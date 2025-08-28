#Adapted from https://github.com/Zheng-Chong/CatVTON/blob/main/model/cloth_masker.py

import os
from PIL import Image
from typing import Union
import numpy as np
import cv2
from diffusers.image_processor import VaeImageProcessor
import torch

from .SCHP import SCHP  # type: ignore
from .DensePose import DensePose  # type: ignore

DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'upper': ['legs'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
}
PROTECT_BODY_PARTS_DENSEPOSE = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['big arms', 'Face'],
    'overall': [],
    'inner': ['legs'],
    'outer': ['legs'],
}
PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'overall': {
        'ATR': [],
        'LIP': []
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    }
}
MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Jumpsuits', 'Dress'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}
MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}
    
schp_public_protect_parts = ['Hat', 'Hair', 'Sunglasses', 'Left-shoe', 'Right-shoe', 'Bag', 'Glove', 'Scarf']
schp_protect_parts = {
    'upper': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits'],  
    'lower': ['Left-arm', 'Right-arm', 'Upper-clothes', 'Coat'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Coat'],
    'outer': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Upper-clothes']
}
schp_mask_parts = {
    'upper': ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits', 'socks'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits', 'socks'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}

dense_mask_parts = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}

def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))

def part_mask_of(part: Union[str, list],
                 parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

def hull_mask(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask
    

def rectangle_mask(mask_area, padding = 0):
    hull_mask = np.zeros_like(mask_area)
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        hull_mask = cv2.rectangle(np.zeros_like(mask_area), (x - padding, y - 10), (x + w + padding, y + h), 255, -1) | hull_mask
    return hull_mask


def find_max_hull(mask_area):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    print(max_area)
    if max_area == 0: return mask_area
    hull = cv2.convexHull(max_contour)
    hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255)
    return hull_mask

def bottom_to_top_erode(mask, erosion_iterations=1, kernel_size=3):
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i >= (kernel_size // 2):
                kernel[i, j] = 1

    eroded_mask = cv2.erode(binary_mask, kernel, iterations=erosion_iterations)

    return eroded_mask
 
class AutoMasker:
    def __init__(
        self, 
        densepose_ckpt='./ckpts/DensePose', 
        schp_ckpt='./ckpts/SCHP', 
        device='cuda'):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        self.densepose_processor = DensePose(densepose_ckpt, device)
        self.schp_processor_atr = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), device=device)
        self.schp_processor_lip = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), device=device)
        
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def process_densepose(self, image_or_path):
        return self.densepose_processor(image_or_path, resize=1024)

    def process_schp_lip(self, image_or_path):
        return self.schp_processor_lip(image_or_path)

    def process_schp_atr(self, image_or_path):
        return self.schp_processor_atr(image_or_path)
        
    def preprocess_image(self, image_or_path, lip=False):
        return {
            'densepose': self.densepose_processor(image_or_path, resize=1024),
            'schp_atr': self.schp_processor_atr(image_or_path),
            'schp_lip': None if not lip else self.schp_processor_lip(image_or_path)
        }
    
    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,
        schp_atr_mask: Image.Image,
        part: str='overall',
        with_bag=False,
        protect_face=False,
        padding = 20,
        **kwargs
    ):
        assert part in ['upper', 'lower', 'overall', 'inner', 'outer'], f"part should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {part}"
        w, h = densepose_mask.size
        
        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        
        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
        
        densepose_mask = np.array(densepose_mask)
        if schp_lip_mask is not None: schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)
        
        # Strong Protect Area (Hands, Face, Accessory, Feet)
        hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        hands_protect_area = hands_protect_area & part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING)
            # (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
            #  part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
        # face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)
        face_protect_area = part_mask_of(['face'], densepose_mask, DENSE_INDEX_MAP) | part_mask_of('Face', schp_atr_mask, ATR_MAPPING) 
        
        ## hands erode
        hands_protect_area = cv2.erode(hands_protect_area * 255, np.ones((3, 3), np.uint8), iterations=3)
        hands_protect_area[hands_protect_area < 25] = 0
        hands_protect_area[hands_protect_area >= 25] = 1
        
        strong_protect_area = hands_protect_area | face_protect_area 

        # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
        body_protect_area = part_mask_of(PROTECT_BODY_PARTS_DENSEPOSE, densepose_mask, DENSE_INDEX_MAP)
        #part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING) # ｜ part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING)
        hair_protect_area = part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING) # ｜ part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING)
        cloth_protect_area = part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING) # ｜ part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING)
        accessory_protect_area = part_mask_of((
            accessory_parts := ['Hat', 'Glove', 'Sunglasses', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks'] + (['Bag'] if with_bag else [])
            ), schp_atr_mask, ATR_MAPPING) 
            #| part_mask_of(accessory_parts, schp_lip_mask, LIP_MAPPING) 
        
        hair_protect_area = hair_protect_area & (~part_mask_of(['big arms'], densepose_mask, DENSE_INDEX_MAP))
        weak_protect_area = body_protect_area | hair_protect_area | strong_protect_area | cloth_protect_area | accessory_protect_area
        weak_protect_area_nocloth = body_protect_area | hair_protect_area | accessory_protect_area
        
        # Mask Area
        strong_mask_area = part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING) #｜ part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING)
        background_area = part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING) #& part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING)
        mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
        # mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), (densepose_mask.shape[1]//2, densepose_mask.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel*2+1, iterations=2)
        # mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), (densepose_mask.shape[1], densepose_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # print(densepose_mask.shape, weak_protect_area.shape, background_area.shape, mask_dense_area.shape)
        mask_area = (np.ones_like(densepose_mask) & (~weak_protect_area) & (~background_area)) | mask_dense_area

        # mask_area = hull_mask(mask_area * 255) // 255  # Convex Hull to expand the mask area
        mask_area = rectangle_mask(mask_area * 255, padding=20) // 255
        mask_area = mask_area & (~weak_protect_area_nocloth)#(~weak_protect_area)
        mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        
        # densepose_face = cv2.erode(part_mask_of(['face'], densepose_mask, DENSE_INDEX_MAP) * 255, np.ones((3, 3), np.uint8), iterations=5)
        # densepose_face[densepose_face < 25] = 0
        # densepose_face[densepose_face >= 25] = 1
        densepose_face = part_mask_of(['face'], densepose_mask, DENSE_INDEX_MAP) & part_mask_of('Face', schp_atr_mask, ATR_MAPPING) & part_mask_of('Face', schp_lip_mask, LIP_MAPPING)
        # densepose_face = bottom_to_top_erode(densepose_face * 255, erosion_iterations=7, kernel_size=3) //255
        
        # cv2.imwrite('./show_vis/densepose_face.jpg', (densepose_face ^ part_mask_of(['face'], densepose_mask, DENSE_INDEX_MAP))*255)
        # print(part_mask_of(['face'], densepose_mask, DENSE_INDEX_MAP).sum(), densepose_face.sum())
        mask_area = (mask_area | strong_mask_area) & ((~hands_protect_area) if not protect_face else (~(hands_protect_area | densepose_face)))
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)

        return Image.fromarray(mask_area * 255)
        
    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
        with_bag = False,
        protect_face=False,
        padding=40
    ):
        assert mask_type in ['upper', 'lower', 'overall', 'inner', 'outer'], f"mask_type should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {mask_type}"
        preprocess_results = self.preprocess_image(image, lip=protect_face)
        mask = self.cloth_agnostic_mask(
            preprocess_results['densepose'], 
            preprocess_results['schp_lip'], 
            preprocess_results['schp_atr'], 
            part=mask_type,
            with_bag=with_bag,
            protect_face=protect_face,
            padding=padding
        )
        return {
            'mask': mask,
            'densepose': preprocess_results['densepose'],
            'schp_lip': None if not protect_face else preprocess_results['schp_lip'],
            'schp_atr': preprocess_results['schp_atr']
        }


if __name__ == '__main__':
    import av
    import os
    from PIL import Image
    from pathlib import Path

    def save_videos_from_pil(pil_images, path, fps=8):
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
    
    masker = AutoMasker()
    
    # p = '/mnt/nas_3d_huakun/wm/tmp/video/tryon/demos/taobao/videos/model_1.mp4' 
    # p = '/mnt/nas_3d_huakun/wm/tmp/video/tryon/demos/inputs/videos/upper1.mp4'
    p = 'test.mp4'
    # p = '/mnt/nas_3d_huakun/wm/tmp/video/4D_Tryon/Taobao/lower_body/videos/FMRP_1647597305856395_vedio_20240731051152_detail-Scene-002.mp4'
    frames = read_frames(p)
    
    res = []
    seg1 = []
    seg2 = []
    pose = []
    for i in frames:
        w,h = i.size
        i.resize((w//4*4, h//4*4), Image.Resampling.LANCZOS)
        tmp = masker(i, mask_type='upper')
        mask = tmp['mask']
        print(mask.size)
        mask = np.array(mask)
        print(mask.shape)
        masked_vton_img = Image.composite(Image.fromarray((mask/255*127).astype(np.uint8)), i, Image.fromarray((mask).astype(np.uint8)))
        res.append(masked_vton_img)
        # seg1.append(tmp['schp_lip'])
        seg2.append(tmp['schp_atr'])
        pose.append(tmp['densepose'])
    
    save_videos_from_pil(res, '../show_vis/upper_mask_2.mp4', fps=24)
    # save_videos_from_pil(seg1, '../show_vis/seg1.mp4', fps=24)
    save_videos_from_pil(seg2, '../show_vis/seg2.mp4', fps=24)
    save_videos_from_pil(pose, '../show_vis/pose.mp4', fps=24)

