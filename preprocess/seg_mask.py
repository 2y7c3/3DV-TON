from argparse import ArgumentParser
import os
from tqdm import tqdm
import cv2
import numpy as np
import av
from PIL import Image
from pathlib import Path
from model.cloth_masker import AutoMasker

def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)
        stream.bit_rate = 5000000  # 比特率: 5 Mbps
        stream.options = {
            'crf': '20',  # Constant Rate Factor, lower values are higher quality
            'preset': 'medium'  # Encoding speed vs quality tradeoff, can be 'ultrafast', 'superfast', 'fast', 'medium', 'slow', 'slower', 'veryslow'
        }
        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output', default=None, help='Input image dir')
    parser.add_argument('--type', help='cloth_type')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    
    parser.add_argument("-dst", type=int, default=0)
    parser.add_argument("-dlen", type=int, default=-1)
    
    parser.add_argument("--bag", action="store_true")
    parser.add_argument("--protect_face", action="store_true")
    
    args = parser.parse_args()
    
    is_img = False
    
    import glob
    
    if is_img:
        files = glob.glob(os.path.join(args.input, '*.jpg'))
    else:    
        if '.mp4' not in args.input:
            files = glob.glob(os.path.join(args.input, '*.mp4'))
        else:
            files = glob.glob(args.input)
    if args.output is not None:
        output_root = args.output
    else:
        output_root = os.path.join('/', *args.input.split('/')[:-1])
    
    masker = AutoMasker()

    for p in tqdm(files[args.dst:(len(files) if args.dlen == -1 else args.dst+args.dlen)]):
        try:
            print(p)
            if is_img:
                name = p.split('/')[-1][:-4]
                save_dir = os.path.join(output_root, 'rec_agnostic', name+'.png')
                save_dir_mask = os.path.join(output_root, 'rec_agnostic_mask',name+'.png')
                os.makedirs(os.path.join(output_root, 'rec_agnostic'), exist_ok=True)
                os.makedirs(os.path.join(output_root, 'rec_agnostic_mask'), exist_ok=True)
                
                if os.path.exists(save_dir_mask):
                    print(save_dir_mask, "exist!!")
                    continue
            
                frames = [Image.open(p).convert("RGB")]
            else:
                name = p.split('/')[-1][:-4]
                save_dir = os.path.join(output_root, 'rec_agnostic', name+'.mp4')
                save_dir_mask = os.path.join(output_root, 'rec_agnostic_mask',name+'.mp4')

                if os.path.exists(save_dir_mask):
                    print(save_dir_mask, "exist!!")
                    continue
                
                frames = read_frames(p)
            
            agn, agn_mask= [],[]
            for i in frames:
                tmp = masker(i, mask_type=args.type, padding=40, with_bag=args.bag, protect_face=args.protect_face)
                mask = tmp['mask']
                masked_vton_img = Image.composite(Image.fromarray((np.array(mask)/255*127).astype(np.uint8)), i, mask)
                agn.append(masked_vton_img)
                agn_mask.append(mask)
            
            if is_img:
                agn[0].save(save_dir)
                agn_mask[0].save(save_dir_mask)
            else:    
                save_videos_from_pil(agn, save_dir, fps=24)
                save_videos_from_pil(agn_mask, save_dir_mask, fps=24)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()
