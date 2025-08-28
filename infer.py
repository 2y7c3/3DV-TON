import argparse
from datetime import datetime
from pathlib import Path
import os
import glob
 
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline

from src.utils.util import get_fps, read_frames, save_videos_grid, save_videos_from_pil
import json

OFFSET=0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="./configs/prompts/test.yaml")
    args = parser.parse_args()

    return args

def resize_and_center_crop(image, desired_size):
    original_width, original_height = image.size
    desired_width, desired_height = desired_size
    
    original_aspect_ratio = original_width / original_height
    desired_aspect_ratio = desired_width / desired_height
    
    if original_aspect_ratio > desired_aspect_ratio:
        new_height = int(desired_height)
        new_width = int(new_height * original_aspect_ratio)
    else:
        new_width = int(desired_width)
        new_height = int(new_width / original_aspect_ratio)
    
    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - desired_width) / 2
    top = (new_height - desired_height) / 2
    right = (new_width + desired_width) / 2
    bottom = (new_height + desired_height) / 2

    return image_resized.crop((left, top, right, bottom))

def infer(
    unet,
    vae,
    ref_unet,
    global_step, 
    config,
    fps,
    save_dir,
    steps=20,
    W=384, 
    H=512, 
    L=24, 
    seed=42, 
    cfg=3.5, 
    use_smpl=False,
    use_mesh=False,
    use_one_frame=False,
    st = 0,
    inference_config_path=None
):
    config = OmegaConf.load(config)

    device = unet.device
    weight_type = unet.dtype
    print(device, weight_type)
    
    with torch.no_grad():
        vae = vae
        reference_unet = ref_unet
        denoising_unet = unet

        inference_config_path = config.inference_config if inference_config_path is None else inference_config_path
        infer_config = OmegaConf.load(inference_config_path)

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        seed = config.get("seed", seed)
        generator = torch.manual_seed(seed)

        width, height = W, H
        print(W,H)
        _clip_length = config.get("L", L)  
        steps = steps
        guidance_scale = cfg

        pipe = Pose2VideoPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )
        pipe = pipe.to(device, dtype=weight_type)

        _save_dir = save_dir
        save_dir = Path(os.path.join(save_dir, 'val'))
        save_dir.mkdir(exist_ok=True, parents=True)
        save_dir_pred = Path(os.path.join(_save_dir, 'pred'))
        save_dir_pred.mkdir(exist_ok=True, parents=True)
        model_video_paths = config.model_video_paths
        cloth_image_paths = config.cloth_image_paths
        print(len(model_video_paths), len(cloth_image_paths))

        if '*' in cloth_image_paths[0]: 
            cloth_image_paths = glob.glob(cloth_image_paths[0])

        transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        for i,model_image_path in enumerate(model_video_paths):
            if len(model_image_path) == 2:
                st, clip_length = model_image_path[1]
                st += (OFFSET) ## NOTE
                clip_length += (OFFSET)
                model_image_path = model_image_path[0]
            else:
                clip_length = None
            
            if not os.path.exists(model_image_path): continue
            
            src_fps = get_fps(model_image_path)
            model_name = Path(model_image_path).stem
            agnostic_path=model_image_path.replace("videos","rec_agnostic")
            agn_mask_path=model_image_path.replace("videos","rec_agnostic_mask")
            smpl_path = model_image_path.replace("videos", "smpl_vids")[:-4] + '_incam.mp4'

            ## prepare
            video_tensor_list=[]
            video_images=read_frames(model_image_path)
            if clip_length == None: clip_length = min(st + _clip_length, len(video_images))
            else: clip_length = min(clip_length, min(st + _clip_length, len(video_images)))
            for vid_image_pil in video_images[st:clip_length]:
                video_tensor_list.append(transform(resize_and_center_crop(vid_image_pil, (width, height))))
            video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
            video_tensor = video_tensor.transpose(0, 1)
            agnostic_list=[]
            agnostic_images=read_frames(agnostic_path)
            for agnostic_image_pil in agnostic_images[st:clip_length]:
                agnostic_list.append(resize_and_center_crop(agnostic_image_pil, (width, height)))
            agn_mask_list=[]
            agn_mask_images=read_frames(agn_mask_path)
            for agn_mask_image_pil in agn_mask_images[st:clip_length]:
                agn_mask_list.append(resize_and_center_crop(agn_mask_image_pil, (width, height)))
            smpl_list=[]
            smpl_images=read_frames(smpl_path)
            for smpl_image_pil in smpl_images[st:clip_length]:
                smpl_list.append(resize_and_center_crop(smpl_image_pil, (width, height)))
            video_tensor = video_tensor.unsqueeze(0)

            for cloth_image_path in (cloth_image_paths[i:i+1] if not config.for_each else cloth_image_paths):
                cloth_name =  Path(cloth_image_path).stem
                cloth_image_pil = resize_and_center_crop(Image.open(cloth_image_path).convert("RGB"), (width, height))


                try:
                    print("!!", os.path.join(config.tryon_res_path, '*' + cloth_name + '*' + model_name + '*.*'))
                    one_frame_p = glob.glob(os.path.join(config.tryon_res_path, '*' + cloth_name + '*' + model_name + '*.*'))[0]
                except:
                    print("~~", os.path.join(config.tryon_res_path, '*' + model_name + '*' + cloth_name + '*.*'))
                    one_frame_p = glob.glob(os.path.join(config.tryon_res_path, '*' + model_name + '*' + cloth_name + '*.*'))[0]
                print(one_frame_p, model_image_path, cloth_image_path)
                one_frame = resize_and_center_crop(Image.open(one_frame_p).convert("RGB"), (width, height))

                try:
                    if not use_mesh:
                        mesh_list = [Image.new("RGB", (width, height), (127, 127, 127)) for _ in range(video_tensor.shape[2])]
                    else:
                        try:
                            _mesh_path = glob.glob(os.path.join(config.mesh_path, '*' + cloth_name + '*' + model_name + '*.*'))[0]
                        except:
                            _mesh_path = glob.glob(os.path.join(config.mesh_path, '*' + model_name + '*' + cloth_name + '*.*'))[0]
                        mesh_list=[]
                        mesh_images=read_frames(_mesh_path)
                        for mesh_image_pil in mesh_images[st:clip_length]:
                            mesh_list.append(resize_and_center_crop(mesh_image_pil, (width, height)))
                except:
                    mesh_list = [Image.new("RGB", (width, height), (127, 127, 127)) for _ in range(video_tensor.shape[2])]
                
                
                if global_step == 0:
                    save_videos_from_pil(agnostic_list, f"{save_dir}/debug_agn_{model_name}.mp4", fps=src_fps if fps is None else fps)
                    save_videos_from_pil(agn_mask_list, f"{save_dir}/debug_agn_mask_{model_name}.mp4", fps=src_fps if fps is None else fps)
                    save_videos_from_pil(mesh_list, f"{save_dir}/debug_mesh_{model_name}_{cloth_name}.mp4", fps=src_fps if fps is None else fps)
                    one_frame.save(f"{save_dir}/debug_tryon_{model_name}_{cloth_name}.jpg")
                    cloth_image_pil.save(f"{save_dir}/debug_cloth_{cloth_name}.jpg")
                    
                pipeline_output = pipe(
                    agnostic_list,
                    agn_mask_list,
                    cloth_image_pil,
                    one_frame if use_one_frame else None,
                    width,
                    height,
                    clip_length - st,
                    steps,
                    guidance_scale,
                    generator=generator,
                    smpl_images = smpl_list,
                    mesh_images = mesh_list,
                    noisy_tryon=True
                )
                video = pipeline_output.videos
                _video = torch.cat([video_tensor,video], dim=0)
                save_videos_grid(
                    _video,
                    f"{save_dir}/{global_step}_{model_name}_{cloth_name}_{H}x{W}_{int(guidance_scale)}.mp4",
                    n_rows=2,
                    fps=src_fps if fps is None else fps,
                )
                
                save_videos_grid(
                    video,
                    f"{save_dir_pred}/{global_step}_{model_name}_{cloth_name}_{H}x{W}_{int(guidance_scale)}.mp4",
                    n_rows=1,
                    fps=src_fps if fps is None else fps,
                )
    
def main(cfg):
    dtype = torch.float16
    
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_vae_path,
    ).to("cuda", dtype=dtype)

    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path,
        subfolder="unet",
        unet_additional_kwargs={
            "in_channels": 4,
            "cross_attention_dim": None
        }
    ).to(dtype=dtype, device="cuda")

    inference_config_path = cfg.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=dtype, device="cuda")

    def load_ckpt(checkpoint_path, model):
        if checkpoint_path != None:
            print(f"from checkpoint: {checkpoint_path}")
            checkpoint_path = torch.load(checkpoint_path, map_location="cpu")
            if "global_step" in checkpoint_path: 
                print(f"global_step: {checkpoint_path['global_step']}")
            state_dict = checkpoint_path["state_dict"] if "state_dict" in checkpoint_path else checkpoint_path

            m, u = model.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0
    load_ckpt(cfg.denoising_unet_path, denoising_unet)
    load_ckpt(cfg.reference_unet_path, reference_unet)
    
    infer(
        denoising_unet,
        vae,
        reference_unet,
        30,
        args.config,
        cfg.fps,
        cfg.save_dir,
        steps=cfg.steps,
        W=cfg.width, 
        H=cfg.height, 
        L=24, 
        seed=42, 
        cfg=2.5, 
        use_smpl=cfg.use_smpl,
        use_mesh=cfg.use_mesh,
        use_one_frame=cfg.one_frame,
        st = 0,
        inference_config_path=inference_config_path
    )
    
if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
