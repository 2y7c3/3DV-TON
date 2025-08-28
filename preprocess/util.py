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
import numpy as np

import imageio
import random
import cv2


def is_image_washed_out(image, brightness_threshold=200, contrast_threshold=20, saturation_threshold=30, bright_pixel_ratio=0.7):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if image is None:
        raise ValueError("无法读取图像")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_brightness = np.mean(v)
    contrast = np.std(v)
    mean_saturation = np.mean(s)
    
    bright_pixels = np.sum(v > brightness_threshold)
    total_pixels = v.size
    bright_ratio = bright_pixels / total_pixels
    
    is_washed_out = (
        mean_brightness > brightness_threshold and
        contrast < contrast_threshold and
        mean_saturation < saturation_threshold and
        bright_ratio > bright_pixel_ratio
    )
    
    return is_washed_out, {
        'mean_brightness': mean_brightness,
        'contrast': contrast,
        'mean_saturation': mean_saturation,
        'bright_pixel_ratio': bright_ratio
    }


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(
            1, 3, 1
        ).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def axis_angle_to_rotation_matrix(axis_angle):
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.

    Args:
        axis_angle: tensor of 3d vector of axis-angle rotations in radians with shape :math:`(N, 3)`.

    Returns:
        tensor of rotation matrices of shape :math:`(N, 3, 3)`.

    Example:
        >>> input = tensor([[0., 0., 0.]])
        >>> axis_angle_to_rotation_matrix(input)
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]])

        >>> input = tensor([[1.5708, 0., 0.]])
        >>> axis_angle_to_rotation_matrix(input)
        tensor([[[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
                 [ 0.0000e+00, -3.6200e-06, -1.0000e+00],
                 [ 0.0000e+00,  1.0000e+00, -3.6200e-06]]])
    """
    # if not isinstance(axis_angle, Tensor):
    #     raise TypeError(f"Input type is not a Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {axis_angle.shape}")

    def _compute_rotation_matrix(axis_angle, theta2, eps: float = 1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the axis_angle vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = axis_angle / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(axis_angle):
        rx, ry, rz = torch.chunk(axis_angle, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _axis_angle = torch.unsqueeze(axis_angle, dim=1)
    theta2 = torch.matmul(_axis_angle, _axis_angle.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(axis_angle, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(axis_angle)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    # create output pose matrix with masked values
    rotation_matrix = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx3x3

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
    startpoints = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
    max_offset = int(0.2 * image_width)
    endpoints = [
        (random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)),
        (image_width + random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)),
        (image_width + random.randint(-max_offset, max_offset), image_height + random.randint(-max_offset, max_offset)),
        (random.randint(-max_offset, max_offset), image_height + random.randint(-max_offset, max_offset))
    ]
    params = {
        'rotate': random.uniform(-5, 5),  
        'affine': {
            'degrees': random.uniform(0, 0),  
            'translate': (random.uniform(-0.0, 0.0), random.uniform(-0.0, 0.0)),  
            'scale': random.uniform(0.8, 1.2), 
            'shear': random.uniform(0, 0), 
        },
        'perspective': {'distortion_scale': random.uniform(0.1, 0.5), "startpoints": startpoints, "endpoints": endpoints},
        'flip': {'horizontal': random.random() < 0.5, 'vertical': random.random() < 0.5},  
        'aspect_ratio': random.uniform(0.8, 1.2),
    }
    return params
