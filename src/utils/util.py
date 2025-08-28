import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

import imageio
import random

import gc

import math
from typing import List, Set, Union

import cv2
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision.io import read_video   

def read_video_frames(
    video_path, 
    start_frame=0, 
    end_frame=None, 
    to_float=True,
    normalize=True
    ):
    """Read video frames from video file.
    Args:
        video_path (str): Path to video file.
        start_frame (int, optional): Start frame index. Defaults to 0.
        end_frame (int, optional): End frame index. Defaults to None.
        to_float (bool, optional): Convert video frames to float32. Defaults to True.
        normalize (bool, optional): Normalize video frames to [-1, 1]. Defaults to True.
    Returns:
        torch.Tensor: Video frames in B(1)CTHW format.
    """
    video = read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
    end_frame = min(end_frame, video.size(0)) if end_frame is not None else video.size(0)
    video = video[start_frame:end_frame].permute(1, 0, 2, 3).unsqueeze(0)
    if to_float:
        video = video.float() / 255.0
    if normalize:
        if to_float:
            video = video * 2 - 1
        else:
            raise ValueError("`to_float` must be True when `normalize` is True")
    return video

def scan_files_in_dir(directory, postfix: Set[str] = None, progress_bar: tqdm = None) -> list:
    """
    Scans files in the specified directory and its subdirectories, 
    and filters files based on their extensions.

    Args:
        directory (str): The path to the directory to be scanned.
        postfix (Set[str], optional): A set of file extensions to filter. Defaults to None, 
            which means no filtering will be performed.
        progress_bar (tqdm, optional): A tqdm progress bar object to display the scanning progress. 
            Defaults to None, which means no progress bar will be displayed.

    Returns:
        list: A list of files found in the directory and its subdirectories, 
            where each file is represented as an os.DirEntry object.
    """
    file_list = []
    progress_bar = tqdm(total=0, desc=f"Scanning {directory}", ncols=100) if progress_bar is None else progress_bar
    for entry in os.scandir(directory):
        if entry.is_file():
            # If no extension filtering is specified, or the current file's extension is in the filter, 
            # add it to the file list
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        elif entry.is_dir():
            # If the current entry is a directory, recursively call this function to scan its subdirectories
            file_list += scan_files_in_dir(entry.path, postfix=postfix, progress_bar=progress_bar)
    return file_list

def reset_memory(device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


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


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


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


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps
                    # h w
def get_points_map(origin_size,target_size,track_path,visibility_path):
    track=torch.load(track_path)
    vis=torch.load(visibility_path)
    
    num_frame=vis.shape[0]
    num_points=vis.shape[1]

    origin_height=origin_size[0]
    origin_width=origin_size[1]

    height=target_size[0]
    width=target_size[1]

    points_list=[]
    for i in range(num_frame):
        frame=track[i]
        points_tensor=torch.zeros((height,width))# check
        for point in range(num_points):
            x,y=frame[point].numpy()
           
            if vis[i][point]:
                scaled_x=round(width*x/origin_width)
                scaled_y=round(height*y/origin_height)

                points_tensor[int(scaled_y)][int(scaled_x)]=1.0
        points_list.append(points_tensor)
   
    return points_list

def get_zero_points_map(clip_length,size):
    points_list=[]
    for i in range(clip_length):
        points_list.append(torch.zeros(size))

    return points_list

def save_videos_grid_dataset(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=None, save_every_image=False, dir_path=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    video_length = videos.shape[0]
    outputs = []
    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        if x.max() <= 1.0:
            x = (x * 255).numpy().astype(np.uint8)
        else:
            x = x.numpy().astype(np.uint8)
        
        outputs.append(x)

    # os.makedirs(os.path.dirname(path), exist_ok=True)
    if fps is None:
        fps = (video_length // 2) if video_length > 1 else 1
    
    if path.endswith('.gif'):
        imageio.mimsave(path, outputs, fps=fps, loop=0)
    else:
        imageio.mimsave(path, outputs, fps=fps, codec='libx264')
    
    if save_every_image:
        dir_base_path = path[:-4]
        os.makedirs(dir_base_path, exist_ok=True)
        for i, x in enumerate(videos):
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            if x.max() <= 1.0:
                x = (x * 255).numpy().astype(np.uint8)
            else:
                x = x.numpy().astype(np.uint8)
            
            Image.fromarray(x).save(f"{dir_base_path}/_{i}.png")

def generate_random_params(image_width, image_height):
    """生成包含随机参数的字典"""
    # 生成起始点（图像的四个角）
    startpoints = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
    max_offset = int(0.2 * image_width)
    # 生成结束点，每个点在原位置基础上加上一个随机偏移
    endpoints = [
        (random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)),
        (image_width + random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)),
        (image_width + random.randint(-max_offset, max_offset), image_height + random.randint(-max_offset, max_offset)),
        (random.randint(-max_offset, max_offset), image_height + random.randint(-max_offset, max_offset))
    ]
    params = {
        'rotate': random.uniform(-5, 5),  # 在-30到30度之间随机选择一个角度
        'affine': {
            'degrees': random.uniform(0, 0),  # 仿射变换的角度，这里设定为-15到15度之间
            'translate': (random.uniform(-0.0, 0.0), random.uniform(-0.0, 0.0)),  # 平移比例
            'scale': random.uniform(0.8, 1.2),  # 缩放比例
            'shear': random.uniform(0, 0),  # 剪切强度
        },
        'perspective': {'distortion_scale': random.uniform(0.1, 0.5), "startpoints": startpoints, "endpoints": endpoints},  # 透视变换强度
        'flip': {'horizontal': random.random() < 0.5, 'vertical': random.random() < 0.5},  # 翻转概率
        'aspect_ratio': random.uniform(0.8, 1.2),  # 宽高比调整
    }
    return params
