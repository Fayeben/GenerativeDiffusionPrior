"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.script_util_x0 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from save_image_utils import save_images
from npz_dataset import NpzDataset, DummyDataset
from imagenet_dataloader.imagenet_dataset import ImageFolderDataset

import torchvision.transforms as transforms
from PIL import Image,ImageFilter
import cv2
import pdb
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlur, self).__init__()
        self.kernel_size = len(kernel)
        # print('kernel size is {0}.'.format(self.kernel_size))
        assert self.kernel_size % 2 == 1, 'kernel size must be odd.'
        
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:,0,:,:].unsqueeze_(1)
        x2 = x[:,1,:,:].unsqueeze_(1)
        x3 = x[:,2,:,:].unsqueeze_(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight)
        x2 = F.conv2d(x2, self.weight)
        x3 = F.conv2d(x3, self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    

def get_gaussian_blur(kernel_size, device):
    kernel = gkern(kernel_size, 2).astype(np.float32)
    gaussian_blur = GaussianBlur(kernel)
    return gaussian_blur.to(device)

def get_dataset(path, global_rank, world_size):
    if os.path.isfile(path): # base_samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        dataset = ImageFolderDataset(path, label_file='./imagenet_dataloader/imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=global_rank, world_size=world_size)
    return dataset

def main():
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = th.device('cuda')
    save_dir = args.save_dir if len(args.save_dir)>0 else None

    # dist_util.setup_dist()
    logger.configure(dir = save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def blur_cond_fn(x, t, y=None, x_lr=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            if not x_lr is None:
                # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'

                device_x_in_lr = x_in.device
                blur = get_gaussian_blur(kernel_size=9, device=device_x_in_lr)
                x_in_lr = blur((x_in+1)/2)

                if sample_noisy_x_lr:
                    t_numpy = t.detach().cpu().numpy()
                    spaced_t_steps = [diffusion.timestep_reverse_map[t_step] for t_step in t_numpy]
                    if sample_noisy_x_lr_t_thred is None or spaced_t_steps[0] < sample_noisy_x_lr_t_thred:
                        print('Sampling noisy lr')
                        spaced_t_steps = th.Tensor(spaced_t_steps).to(t.device).to(t.dtype)
                        x_lr = diffusion.q_sample(x_lr, spaced_t_steps)

                x_lr = (x_lr + 1) / 2
                mse = (x_in_lr - x_lr) ** 2
                mse = mse.mean(dim=(1,2,3))
                mse = mse.sum()
                loss = - mse * args.img_guidance_scale # move xt toward the gradient direction
                print('step t %d img guidance has been used, mse is %.8f * %d = %.2f' % (t[0], mse, args.img_guidance_scale, mse*args.img_guidance_scale))
            return th.autograd.grad(loss, x_in)[0]

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("loading dataset...")
    # load gan or vae generated images
    if args.start_from_scratch and args.use_img_for_guidance:
        pass
    else:
        if args.start_from_scratch:
            dataset = DummyDataset(args.num_samples, rank=args.global_rank, world_size=args.world_size)
        else:
            dataset = get_dataset(args.dataset_path, args.global_rank, args.world_size)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # load lr images that are used for guidance 
    if args.use_img_for_guidance:
        dataset_lr = get_dataset(args.base_samples, args.global_rank, args.world_size)
        dataloader_lr = th.utils.data.DataLoader(dataset_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)
        if args.start_from_scratch:
            dataset = DummyDataset(len(dataset_lr), rank=0, world_size=1)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader = zip(dataloader, dataloader_lr)

    # args.save_png_files=True
    if args.save_png_files:
        print(logger.get_dir())
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'lr'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'gt'), exist_ok=True)
        start_idx = args.global_rank * dataset.num_samples_per_rank

    logger.log("sampling...")
    all_images = []
    all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    for i, data in enumerate(dataloader):
        if args.use_img_for_guidance:
            image, label = data[0]
            image_lr, label = data[1]
            image_lr = image_lr.to(device)
            cond_fn = lambda x,t,y : blur_cond_fn(x, t, y=y, x_lr=image_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
        else:
            image, label = data
            cond_fn = lambda x,t,y : general_cond_fn(x, t, y=y, x_lr=None)
        if args.start_from_scratch:
            shape = (image.shape[0], 3, args.image_size, args.image_size)
        else:
            shape = list(image.shape)
        if args.start_from_scratch and not args.use_img_for_guidance:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(shape[0],), device=device)
        else:
            classes = label.to(device).long()
        image = image.to(device)
        model_kwargs = {}
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.start_from_scratch:
            sample = sample_fn(
                model_fn,
                shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device
            )
        else:
            sample = sample_fn(
                model_fn,
                shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                noise=image,
                denoise_steps=args.denoise_steps
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        image_lr = ((image_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_lr = image_lr.permute(0, 2, 3, 1)
        image_lr = image_lr.contiguous()
        
        sample = sample.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        image_lr = image_lr.detach().cpu().numpy()
        if args.save_png_files:
            save_images(sample, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'images'))
        save_images(image_lr, label.long(), start_idx + len(all_images) * args.batch_size, save_dir=os.path.join(logger.get_dir(), 'lr'))
        all_images.append(sample)
        all_labels.append(classes)
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    if args.save_numpy_array:
        arr = np.concatenate(all_images, axis=0)
        # arr = arr[: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        # label_arr = label_arr[: args.num_samples]

        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_rank_{args.global_rank}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    # dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=False,
        model_path="/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/models/256x256_diffusion_uncond.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # add zhaoyang own's arguments
    parser.add_argument("--device", default=0, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=0, type=int, help='global rank of this process')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    parser.add_argument("--save_dir", default='/mnt/petrelfs/feiben/GDP/generate_images/generated_image_x0_blur_text', type=str, help='the directory to save the generate images')
    parser.add_argument("--save_png_files", action='store_true', help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    # these two arguments are only valid when not start from scratch
    parser.add_argument("--denoise_steps", default=25, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path", default='/mnt/lustre/feiben/DDPM_Beat_GAN/evaluations/precomputed/biggan_deep_imagenet64.npz', type=str, help='path to the generated images. Could be an npz file or an image folder')
    
    parser.add_argument("--use_img_for_guidance", action='store_true', help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')
    parser.add_argument("--img_guidance_scale", default=6000, type=float, help='guidance scale')
    parser.add_argument("--base_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/blur_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--sample_noisy_x_lr", action='store_true', help='whether to first sample a noisy x_lr, then use it for guidance. ')
    parser.add_argument("--sample_noisy_x_lr_t_thred", default=1e8, type=int, help='only for t lower than sample_noisy_x_lr_t_thred, we add noise to lr')
    
    parser.add_argument("--start_from_scratch", action='store_true', help='whether to generate images purely from scratch, not use gan or vae generated samples')
    # num_samples is defined elsewhere, num_samples is only valid when start_from_scratch and not use img as guidance
    # if use img as guidance, num_samples will be set to num of guidance images
    # parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true')
    return parser

import pdb
if __name__ == "__main__":
    main()
