
[[Paper]]() [[Models]](#pretrained-models)

This repository contains PyTorch implementation for __Generative Diffusion Prior for Unified Image Restoration and Enhancement__.


![intro](figs/teaser.png)

This is the repo is based on the open-source repo for [Guided Diffusion](https://github.com/openai/guided-diffusion).
Run the following command the install the guided-diffusion package:
```
pip install -e .
```


# Download Checkpoints and Data

Download pretrained DDPMs on ImageNet-64, LSUN-Bedroom-256, LSUN-Cat-256 from [this page](https://github.com/openai/guided-diffusion). 
Then download pre-generated ImageNet-64 (BigGAN-deep),  LSUN-Bedroom (StyleGAN) and LSUN-Cat (StyleGAN2) images from this page [this page](https://github.com/openai/guided-diffusion/tree/main/evaluations).

# ES-DDPMs Generate Random Images

```
cd scripts
```

Run the following command to use ES-DDPM (T'=100) to generate ImageNet-64 images (with jumping interval 4):
```
python generate_processes_diffusion_and_reverse --execute --reverse_steps 25 --chain_length 250 --dataset imagenet --dataset_path ../evaluations/precomputed/biggan_deep_trunc1_imagenet256.npz --devices '0,1,2,3,4,5,6,7,8' 
```

Run the following command to use ES-DDPM (T'=100) to generate LSUN-Bedroom-256 images:
```
python generate_processes_diffusion_and_reverse --execute --reverse_steps 100 --chain_length 1000 --dataset lsun_bedroom --dataset_path ../evaluations/precomputed/lsun/stylegan_lsun_bedroom.npz --devices '0,1,2,3,4,5,6,7,8' 
```

Run the following command to use ES-DDPM (T'=100) to generate LSUN-Cat-256 images:
```
python generate_processes_diffusion_and_reverse --execute --reverse_steps 100 --chain_length 1000 --dataset lsun_cat --dataset_path ../evaluations/precomputed/lsun/stylegan2_lsun_cat.npz --devices '0,1,2,3,4,5,6,7,8' 
```
