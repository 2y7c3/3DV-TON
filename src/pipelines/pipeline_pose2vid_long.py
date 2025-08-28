# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method

def rescale_cfg(pos, neg, guidance_scale, rescale=0.7):
    cfg = neg + guidance_scale * (pos - neg)
    std_pos = pos.std([1,3,4], keepdim=True) # b c f h w
    std_neg = neg.std([1,3,4], keepdim=True)
    factor = std_pos / std_neg
    factor = factor * rescale + (1-rescale)
    return cfg * factor

@dataclass
class Pose2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class Pose2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.ref_image_processor = VaeImageProcessor(
            do_normalize = True,
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    @torch.no_grad()
    def __call__(
        self,
        agnostic,
        agnostic_mask,
        cloth,
        tryon,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=32,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        smpl_images = None,
        mesh_images = None,
        noisy_tryon=False,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False, #do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False, #do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = 4 #self.denoising_unet.in_channels-5
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            self.vae.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        cloth_image_tensor = self.ref_image_processor.preprocess(
            cloth, height=height, width=width
        )  # (bs, c, width, height)
        cloth_image_tensor = cloth_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        cloth_image_latents = self.vae.encode(cloth_image_tensor).latent_dist.sample()
        cloth_image_latents = cloth_image_latents * self.vae.config.scaling_factor  # (b, 4, h, w)

        if tryon is not None:
            tryon_image_tensor = self.ref_image_processor.preprocess(
                tryon, height=height, width=width
            )  # (bs, c, width, height)
            tryon_image_tensor = tryon_image_tensor.to(
                dtype=self.vae.dtype, device=self.vae.device
            )
            
            if noisy_tryon:
                image_noise_sigma = torch.ones(1).to(device=self.vae.device) * -2#torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.vae.device)
                image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=self.vae.dtype)
                tryon_image_tensor = tryon_image_tensor + torch.randn_like(tryon_image_tensor) * image_noise_sigma[:, None, None, None]
                                
            tryon_image_latents = self.vae.encode(tryon_image_tensor).latent_dist.sample()
            tryon_image_latents = tryon_image_latents * self.vae.config.scaling_factor  # (b, 4, h, w)
        else:
            tryon_image_latents = torch.zeros_like(cloth_image_latents)
        
        # Prepare agnostic
        agn_tensor_list=[]
        for agn in agnostic:
            agn_tensor=self.ref_image_processor.preprocess(
                agn, height=height, width=width
            )  # (1, 3, 512, 384) [-1 1]

            agn_tensor_list.append(agn_tensor)
        agn_tensor = torch.cat(agn_tensor_list, dim=0)  # ((b f), c, h, w)
        agn_tensor = agn_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        agnostic_image_latents = self.vae.encode(agn_tensor).latent_dist.sample()
        agnostic_image_latents = agnostic_image_latents * self.vae.config.scaling_factor  # (b, 4, h, w)
        agnostic_image_latents = rearrange(agnostic_image_latents,"(b f) c h w -> b c f h w",f=video_length)

        # Prepare agnostic mask
        agn_mask_list=[]
        for mask in agnostic_mask:
            mask = self.ref_image_processor.preprocess(
                mask, height=height, width=width
            )  # (bs, c, width, height)
            mask = self.ref_image_processor.denormalize(mask)
            mask = self.ref_image_processor.binarize(mask)
            mask = mask[:,0:1,:,:]
            agn_mask_list.append(mask)
        agn_mask=torch.cat(agn_mask_list,dim=0) # (f c h w)
        agn_mask = torch.nn.functional.interpolate(agn_mask, size=(agn_mask.shape[-2] // 8, agn_mask.shape[-1] // 8))
        agn_mask=rearrange(agn_mask,"(b f) c h w -> b c f h w",f=video_length)

        latents = latents.to(dtype=self.denoising_unet.dtype)
        agnostic_image_latents = agnostic_image_latents.to(dtype=self.denoising_unet.dtype)
        agn_mask = agn_mask.to(dtype=self.denoising_unet.dtype, device=latents.device)

        if smpl_images is not None:
            smpl_cond_tensor_list = []
            for smpl_image in smpl_images:
                smpl_cond_tensor = self.ref_image_processor.preprocess(
                    smpl_image, height=height, width=width
                )

                smpl_cond_tensor_list.append(smpl_cond_tensor)

            smpl_cond_tensor = torch.cat(smpl_cond_tensor_list, dim=0)
            smpl_cond_tensor = smpl_cond_tensor.to(
                device=device, dtype=self.vae.dtype
            )
            smpl_cond_tensor = self.vae.encode(smpl_cond_tensor).latent_dist.sample()
            smpl_cond_tensor = smpl_cond_tensor * self.vae.config.scaling_factor  # (b, 4, h, w) 
            smpl_cond_tensor = rearrange(smpl_cond_tensor,"(b f) c h w -> b c f h w",f=video_length)
            
        if mesh_images is not None:
            mesh_latent_tensor_list = []
            mesh_cond_tensor_list = []
            for mesh_image in mesh_images:
                mesh_image_tensor = self.ref_image_processor.preprocess(
                    mesh_image, height=height, width=width
                )
                mesh_latent_tensor_list.append(mesh_image_tensor)

            mesh_latent_tensor = torch.cat(mesh_latent_tensor_list, dim=0)
            mesh_latent_tensor = mesh_latent_tensor.to(
                device=device, dtype=self.vae.dtype
            )
            mesh_image_latents = self.vae.encode(mesh_latent_tensor).latent_dist.sample()
            mesh_image_latents = mesh_image_latents * self.vae.config.scaling_factor  # (b, 4, h, w) 
            mesh_image_latents = rearrange(mesh_image_latents,"(b f) c h w -> b c f h w",f=video_length)

        pose_fea = torch.cat([smpl_cond_tensor, mesh_image_latents], dim=1)

        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                if i == 0: 
                    if do_classifier_free_guidance:
                        ref_image_latents=torch.cat([
                            torch.zeros_like(cloth_image_latents), cloth_image_latents,
                            torch.zeros_like(tryon_image_latents), tryon_image_latents
                        ], dim=0)
                    else:
                        ref_image_latents=torch.cat([cloth_image_latents, tryon_image_latents], dim=0)
                    
                    self.reference_unet(
                        ref_image_latents,
                        # .repeat(
                        #     (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        # ),
                        torch.zeros_like(t).repeat(ref_image_latents.shape[0]),
                        # t,
                        encoder_hidden_states=None,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        0,
                    )
                )
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = []
                for i in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            i * context_batch_size : (i + 1) * context_batch_size
                        ]
                    )

                latents_cat = torch.cat([latents,agnostic_image_latents,agn_mask, pose_fea], dim=1)
                
                for context in global_context:
                    # 3.1 expand the latents if we are doing classifier free guidance

                    latent_model_input = (
                        torch.cat([latents_cat[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    #noise_pred = rescale_cfg(noise_pred_text, noise_pred_uncond, guidance_scale)

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                
            reference_control_reader.clear()
            reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)
