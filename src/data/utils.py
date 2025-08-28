import gc
import math
import os
import random
from typing import List, Set, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision.io import read_video   

def densepose_to_rgb(densepose: Union[Image.Image, np.ndarray], colormap=cv2.COLORMAP_VIRIDIS):
    """Convert densepose image to RGB image using 
    cv2.COLORMAP_PARULA  # background is black
    cv2.COLORMAP_VIRIDIS # background is purple
    Args:
        densepose (Union[Image.Image, np.ndarray]): Densepose image in L mode.
    Returns:
        PIL.Image.Image: Image in RGB mode.
    """
    if isinstance(densepose, Image.Image):
        assert densepose.mode == 'L', "densepose image must be in L mode."
        densepose = np.array(densepose)
    if densepose.max() <= 24:
        densepose = (densepose / 24.0 * 255.0).astype(np.uint8)
    densepose_bgr = cv2.applyColorMap(densepose, colormap=colormap)
    densepose_rgb = cv2.cvtColor(densepose_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(densepose_rgb)

def densepose_processor(
    densepose_image: Image.Image, 
    height: int = 512, 
    width: int = 384, 
    to_rgb: bool = True, 
    do_normalize: bool = False
):
    """Process densepose image to tensor.

    Args:
        densepose_image (Image.Image): Densepose image in L mode.
        height (int, optional): Defaults to 512.
        width (int, optional): Defaults to 384.
        to_rgb (bool, optional): Defaults to False.
        do_normalize (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: Densepose tensor.
    """
    assert densepose_image.mode == 'L', "densepose image must be in L mode."
    densepose_image = densepose_image.resize((width, height), Image.NEAREST)
    densepose_np = (np.array(densepose_image) / 24.0 * 255.0).astype(np.uint8)
    if to_rgb:
        densepose_image = densepose_to_rgb(densepose_np)
    densepose_tensor = F.to_tensor(densepose_image)
    if do_normalize:
        densepose_tensor = (densepose_tensor - 0.5) / 0.5
    return densepose_tensor

def get_random_mask(shape):
    f, c, h, w = shape
    if f != 1:
        mask_index = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.3, 0.3, 0.3, 0.05])
    else:
        mask_index = np.random.choice([0, 1], p=[0.2, 0.8])
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)
    if mask_index == 0:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()
        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)
        mask[:, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 1:
        mask[:, :, :, :] = 1
    elif mask_index == 2:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:, :, :, :] = 1
    elif mask_index == 3:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
    elif mask_index == 4:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()
        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)
        mask_frame_before = np.random.randint(0, f // 2)
        mask_frame_after = np.random.randint(f // 2, f)
        mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
    else:
        raise ValueError(f"The mask_index {mask_index} is not defined")
    return mask

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

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

# 中心裁剪并缩放帧
def center_crop_and_resize_frame(frame, size):
    target_w, target_h = size
    target_ratio = target_h / target_w  # 目标长宽比
    h, w, _ = frame.shape
    # 计算缩放后的尺寸
    if h / w > target_ratio:
        # 更高，按照宽度缩放
        new_w = target_w
        new_h = int(h * target_w / w)
    else:
        # 更宽，按照高度缩放
        new_h = target_h
        new_w = int(w * target_h / h)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    # 居中裁剪
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    resized_frame = resized_frame[top:top + target_h, left:left + target_w]
    return resized_frame

# 中心裁剪并缩放图像
def center_crop_and_resize_image(image: Image, size):
    target_w, target_h = size
    target_ratio = target_w / target_h  # 目标长宽比
    w, h = image.size
    # 计算缩放后的尺寸
    if w / h > target_ratio:
        # 更高，按照宽度缩放
        new_w = target_w
        new_h = int(target_w * h / w)
    else:
        # 更宽，按照高度缩放
        new_h = target_h
        new_w = int(target_h * w / h)
    resized_image = image.resize((new_w, new_h))
    # 居中裁剪
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    resized_image = image.crop((left, top, left + target_w, top + target_h))
    return resized_image


# 必须是16的倍数，否则经过扩散后会出现维度不匹配的问题
def center_crop_image(image: Image, min_unit=16):
    w, h = image.size
    w_, h_ = w // min_unit * min_unit, h // min_unit * min_unit
    left = (w - w_) // 2
    top = (h - h_) // 2
    right = left + w_
    bottom = top + h_
    image = image.crop((left, top, right, bottom))
    return image


from torchvision.io import write_video
def save_video_frames(frames: torch.Tensor, save_path, fps=24):
    if isinstance(frames, torch.Tensor):
        frames = (frames.permute(1, 2, 3, 0).cpu() + 1.0) / 2.0 * 255.0
        frames = frames.to(torch.uint8)
    write_video(save_path, frames, fps=fps)
    # height, width = frames.shape[1:3]
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    # for frame in frames:
    #     video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # video_writer.release()
    
def get_white_mask(width, height, white_mask_edge=0.1):
    np_mask = np.ones((height, width), dtype=np.uint8) * 255
    if white_mask_edge > 0:
        np_mask[:int(white_mask_edge * height), :] = 0
        np_mask[-int(white_mask_edge * height):, :] = 0
        np_mask[:, :int(white_mask_edge * width)] = 0
        np_mask[:, -int(white_mask_edge * width):] = 0
    return Image.fromarray(np_mask)

def random_mask_aug(mask_image: Image.Image, num_sides=(3, 8)):
    width, height = mask_image.size
    # Construct Random Circle \ Polygon 
    draw = ImageDraw.Draw(mask_image)
    mask_type = random.choice(['circle', 'polygon'])
    # 0. Random Center Point (Not too close to the edge)
    center_point = (random.randint(width//4, width*3//4), random.randint(height//4, height*3//4))
    # 1. Random Radius
    max_radius = min(*center_point, width-center_point[0], height-center_point[1])
    min_radius = max_radius // 4
    # 2. Draw Mask
    if mask_type == 'circle':
        random_box = (
            center_point[0] - random.randint(min_radius, max_radius),
            center_point[1] - random.randint(min_radius, max_radius),
            center_point[0] + random.randint(min_radius, max_radius),
            center_point[1] + random.randint(min_radius, max_radius)
         )
        draw.ellipse(random_box, fill=255)
    elif mask_type == 'polygon':
        num_sides = random.randint(*num_sides)
        radius = random.randint(min_radius, max_radius)
        # 0～ 360 degree select num_sides random points
        angles = [random.uniform(0, 360) for _ in range(num_sides)]
        angles.sort()
        vertices = [
            (center_point[0] + radius * math.cos(math.radians(angle)), center_point[1] + radius * math.sin(math.radians(angle))) 
            for angle in angles
        ]
        draw.polygon(vertices, fill=255)
    return mask_image

def random_crop_resize_with_ratio(
    images: Union[Image.Image, List[Image.Image]],
    scale_range: tuple = (0.8, 1.2), 
    size: tuple = (768, 1024),
):
    """Random crop and resize images with the same ratio (width / height).

    Args:
        images (Union[Image.Image, List[Image.Image]]): _description_
        scale_range (tuple, optional): ratio range. Defaults to (0.8, 1.2).
        size (tuple, optional): return size. Defaults to (768, 1024).

    Returns:
        Union[Image.Image, List[Image.Image]]: resized images
    """
    if is_image := isinstance(images, Image.Image):
        images = [images]
    scale = random.uniform(*scale_range) if scale_range is not None else 1.0
    
    width, height = images[0].size
    if width / height > (ratio := size[0] / size[1]):
        max_width = int(height * ratio)
        max_height = height
    else:
        max_width = width
        max_height = int(width / ratio)
    
    target_height = int(scale * max_height)
    target_width = int(scale * max_width)
    results = []
    if scale <= 1:
        left = random.randint(0, max(0, width - target_width))
        top = random.randint(0, max(0, height - target_height))
        for image in images:
            image = image.crop((left, top, left + target_width, top + target_height))
            image = image.resize((size[0], size[1]), Image.LANCZOS if image.mode == 'RGB' else Image.NEAREST)
            results.append(image)
    elif scale > 1:
        left = random.randint(0, max(0, target_width - width))
        top = random.randint(0, max(0, target_height - height))
        for image in images:
            if image.mode == 'RGB':
                background = Image.new('RGB', (target_width, target_height), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))  # image with random color
            else:
                background = Image.new('L', (target_width, target_height), 'black')
            background.paste(image, (left, top))
            background = background.resize((size[0], size[1]), Image.LANCZOS if image.mode == 'RGB' else Image.NEAREST)
            results.append(background)

    return results if not is_image else results[0]

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


if __name__ == "__main__":
    pass
    
