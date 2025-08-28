# 3DV-TON: Textured 3D-Guided Consistent Video Try-on via Diffusion Models
Min Wei, Chaohui Yu, Jingkai Zhou, and Fan Wang. 2025.
3DV-TON: Textured 3D-Guided Consistent Video Try-on via Diffusion Models. 
In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25),
October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3746027.3754754

[![arXiv](https://img.shields.io/badge/arXiv-2504.17414-b31b1b.svg)](https://arxiv.org/abs/2504.17414)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://2y7c3.github.io/3DV-TON/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/2y7c3/3DV-TON)
[![HR-VVT](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HR_VVT-yellow)](https://huggingface.co/datasets/2y7c3/HR-VVT)

## Installation

```
git clone https://github.com/2y7c3/3DV-TON.git
cd 3DV-TON
pip install -r requirements.txt

cd preprocess/model/DensePose/detectron2/projects/DensePose
pip install -e .

## install GVHMR
## see https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md
## replace GVHMR/hmr4d/utils/vis/renderer.py with our preprocess/renderer.py
```

### Weights

Download [Stable Diffusion](https://huggingface.co/lambdalabs/sd-image-variations-diffusers), [Motion module](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt),[VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) and Our [3DV-TON models](https://huggingface.co/2y7c3/3DV-TON) in ``` ./ckpts ```.

Download [Cloth masker](https://huggingface.co/2y7c3/3DV-TON) in ``` ./preprocess/ckpts ```. Then you can use our cloth masker to generate agnostic mask videos for improved try-on results.

## Inference
We provid three demo examples in ```./demos/``` — run the following commands to test them.

```bash
python infer.py --config ./configs/inference/demo_test.yaml
```

Or you can prepare your own example by following the steps below.

``` bash
# 1. generate agnostic mask (type: 'upper', 'lower', 'overall')
cd preprocess
python seg_mask.py --input demos/videos/video.mp4 --output demos/ --type overall

# 2. use GVHMR to generate SMPL video

# 3. use image tryon model to generate tryon image (e.g. CaTVTON)

# 4. generate textured 3d mesh

# 5. modify demo_test.yaml, then run
python infer.py --config ./configs/inference/demo_test.yaml
```

## BibTeX
```text
@article{wei20253dv,
  title={3dv-ton: Textured 3d-guided consistent video try-on via diffusion models},
  author={Wei, Min and Yu, Chaohui and Zhou, Jingkai and Wang, Fan},
  journal={arXiv preprint arXiv:2504.17414},
  year={2025}
}
```
