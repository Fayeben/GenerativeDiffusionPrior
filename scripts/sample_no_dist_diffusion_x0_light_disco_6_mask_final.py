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
from guided_diffusion.script_util_x0_variance_disco_final import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_model_and_diffusion_final,
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

import MyLoss

def get_dataset(path, global_rank, world_size):
    if os.path.isfile(path): # base_samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        dataset = ImageFolderDataset(path, label_file='./imagenet_dataloader/imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=global_rank, world_size=world_size)
    return dataset

def gray_dgp(image):
    n = int(image.shape[0])
    h = int(image.shape[2])
    w = int(image.shape[3])
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    image_res = gray.view(n, 1, h, w).expand(n, 3, h, w)
    return image_res

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
    model_right, diffusion_right = create_model_and_diffusion_final(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    # args.model_path = "/home/feiben/DDPM_Beat_GAN/256x256_diffusion.pt"
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    # model.to(dist_util.dev())
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    model_right.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    # model.to(dist_util.dev())
    model_right.to(device)
    if args.use_fp16:
        model_right.convert_to_fp16()
    model_right.eval()

    L_exp = MyLoss.L_exp(8,0.3)
    L_color = MyLoss.L_color()
    L_TV = MyLoss.L_TV()

    def color_cond_fn(x, t, light_factor = None, light_variance = None,  y=None, x_lr=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
        assert y is not None
        assert light_factor is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = 0
            if not x_lr is None:
                # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'
                os.makedirs(os.path.join(logger.get_dir(), 'inter'), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), 'inter_gray'), exist_ok=True)
                # if t <1000 and t%100 ==0:
                #     x_in_tmp = x_in
                #     x_in_tmp = ((x_in_tmp + 1) * 127.5).clamp(0, 255).to(th.uint8)
                #     x_in_tmp = x_in_tmp.permute(0, 2, 3, 1)
                #     x_in_tmp = x_in_tmp.contiguous()
                #     x_in_tmp = x_in_tmp.detach().cpu().numpy()
                #     save_images(x_in_tmp, th.Tensor([1]).cuda(), int(t), os.path.join(logger.get_dir(), 'inter'))
                
                # blur = transforms.Compose([transforms.ToPILImage(), Addblur(p=1,blur="Gaussian"), transforms.ToTensor()])
                device_x_in_lr = x_in.device
                x_in_lr = x_in
                light_factor.requires_grad_()
                light_variance.requires_grad_()
                x_in_lr =  (x_in_lr+1)/2 * light_factor + light_variance

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
                loss_exp = torch.mean(L_exp(x_in))
                loss_col = torch.mean(L_color(x_in))
                Loss_TV = L_TV(light_variance)
                loss = loss - mse * args.img_guidance_scale - loss_exp * args.img_guidance_scale / 100 - loss_col * args.img_guidance_scale /200  - Loss_TV * args.img_guidance_scale # move xt toward the gradient direction
                # loss = loss - mse * args.img_guidance_scale 
                # light_factor = light_factor - th.autograd.grad(mse, light_factor,retain_graph=True)[0]
                # light_variance = light_variance - th.autograd.grad(mse, light_variance,retain_graph=True)[0]
                # light_variance = light_variance - th.autograd.grad(loss, light_variance,retain_graph=True)[0]
                print('step t %d img guidance has been used, mse is %.8f * %d = %.2f' % (t[0], mse, args.img_guidance_scale, mse*args.img_guidance_scale))
                # light_factor -= th.autograd.grad(loss, light_factor)[0]
            return light_factor, light_variance, th.autograd.grad(loss, x_in)[0]

    def color_cond_fn_left(x, t, light_factor = None, light_variance = None,  y=None, x_lr=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
        assert y is not None
        assert light_factor is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = 0
            if not x_lr is None:
                # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'
                os.makedirs(os.path.join(logger.get_dir(), 'inter'), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), 'inter_gray'), exist_ok=True)
                # if t <1000 and t%100 ==0:
                #     x_in_tmp = x_in
                #     x_in_tmp = ((x_in_tmp + 1) * 127.5).clamp(0, 255).to(th.uint8)
                #     x_in_tmp = x_in_tmp.permute(0, 2, 3, 1)
                #     x_in_tmp = x_in_tmp.contiguous()
                #     x_in_tmp = x_in_tmp.detach().cpu().numpy()
                #     save_images(x_in_tmp, th.Tensor([1]).cuda(), int(t), os.path.join(logger.get_dir(), 'inter'))

                device_x_in_lr = x_in.device

                x_in_lr = x_in
                light_factor.requires_grad_()
                light_variance.requires_grad_()
                x_in_lr = (x_in_lr+1)/2 * light_factor + light_variance

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
                loss_exp = torch.mean(L_exp(x_in))
                loss_col = torch.mean(L_color(x_in))
                # light_variance_tmp = light_variance[0:256, 0:256]
                Loss_TV = L_TV(light_variance)
                # loss = loss - mse * args.img_guidance_scale - Loss_TV * args.img_guidance_scale/100
                loss = loss - mse * args.img_guidance_scale - loss_exp * args.img_guidance_scale / 100 - loss_col * args.img_guidance_scale /200  - Loss_TV * args.img_guidance_scale # move xt toward the gradient direction
                # loss = loss - mse * args.img_guidance_scale
                light_factor = light_factor - th.autograd.grad(loss, light_factor,retain_graph=True)[0]
                light_variance = light_variance - th.autograd.grad(loss, light_variance,retain_graph=True)[0]
                # light_variance = light_variance - th.autograd.grad(loss, light_variance,retain_graph=True)[0]
                print('step t %d img guidance has been used, mse is %.8f * %d = %.2f' % (t[0], mse, args.img_guidance_scale, mse*args.img_guidance_scale))
                # light_factor -= th.autograd.grad(loss, light_factor)[0]
                # light_variance[0:256, 0:256] = light_variance_tmp
            return light_factor, light_variance, th.autograd.grad(loss, x_in)[0]


    def color_cond_fn_right(x, t, light_factor = None, light_variance = None,  y=None, x_lr=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
        assert y is not None
        assert light_factor is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = 0
            if not x_lr is None:
                # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'
                os.makedirs(os.path.join(logger.get_dir(), 'inter'), exist_ok=True)
                os.makedirs(os.path.join(logger.get_dir(), 'inter_gray'), exist_ok=True)
                # if t <1000 and t%100 ==0:
                #     x_in_tmp = x_in
                #     x_in_tmp = ((x_in_tmp + 1) * 127.5).clamp(0, 255).to(th.uint8)
                #     x_in_tmp = x_in_tmp.permute(0, 2, 3, 1)
                #     x_in_tmp = x_in_tmp.contiguous()
                #     x_in_tmp = x_in_tmp.detach().cpu().numpy()
                #     save_images(x_in_tmp, th.Tensor([1]).cuda(), int(t), os.path.join(logger.get_dir(), 'inter'))

                device_x_in_lr = x_in.device

                x_in_lr = x_in
                light_factor.requires_grad_()
                light_variance.requires_grad_()
                x_in_lr = (x_in_lr+1)/2 * light_factor + light_variance

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
                loss_exp = torch.mean(L_exp(x_in))
                loss_col = torch.mean(L_color(x_in))
                # light_variance_tmp = light_variance[0:256, 128:384]
                Loss_TV = L_TV(light_variance)
                # loss = loss - mse * args.img_guidance_scale - Loss_TV * args.img_guidance_scale/100
                loss = loss - mse * args.img_guidance_scale - loss_exp * args.img_guidance_scale / 100 - loss_col * args.img_guidance_scale /200  - Loss_TV * args.img_guidance_scale # move xt toward the gradient direction
                # loss = loss - mse * args.img_guidance_scale
                light_factor = light_factor - th.autograd.grad(loss, light_factor,retain_graph=True)[0]
                light_variance = light_variance - th.autograd.grad(loss, light_variance,retain_graph=True)[0]
                # light_variance = light_variance - th.autograd.grad(loss, light_variance,retain_graph=True)[0]
                print('step t %d img guidance has been used, mse is %.8f * %d = %.2f' % (t[0], mse, args.img_guidance_scale, mse*args.img_guidance_scale))
                # light_factor -= th.autograd.grad(loss, light_factor)[0]
                # light_variance[0:256, 128:384] = light_variance_tmp
            return light_factor, light_variance, th.autograd.grad(loss, x_in)[0]

    def model_fn(x, t, y=None):
        assert y is not None
        # assert light_factor is not None
        return model(x, t, y if args.class_cond else None)

    def model_fn_right(x, t, y=None):
        assert y is not None
        # assert light_factor is not None
        return model_right(x, t, y if args.class_cond else None)

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

    # print(args.use_img_for_guidance)
    # load lr images that are used for guidance 
    if args.use_img_for_guidance:
        dataset_lefttop_lr = get_dataset(args.base_lefttop_samples, args.global_rank, args.world_size)
        dataset_midtop_lr = get_dataset(args.base_midtop_samples, args.global_rank, args.world_size)
        dataset_righttop_lr = get_dataset(args.base_righttop_samples, args.global_rank, args.world_size)
        dataset_leftbottom_lr = get_dataset(args.base_leftbottom_samples, args.global_rank, args.world_size)
        dataset_midbottom_lr = get_dataset(args.base_midbottom_samples, args.global_rank, args.world_size)
        dataset_rightbottom_lr = get_dataset(args.base_rightbottom_samples, args.global_rank, args.world_size)
        dataset_lr = get_dataset(args.base_samples, args.global_rank, args.world_size)
        dataset_left_lr = get_dataset(args.base_left_samples, args.global_rank, args.world_size)
        dataloader_lefttop_lr = th.utils.data.DataLoader(dataset_lefttop_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader_midtop_lr = th.utils.data.DataLoader(dataset_midtop_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)        
        dataloader_righttop_lr = th.utils.data.DataLoader(dataset_righttop_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader_leftbottom_lr = th.utils.data.DataLoader(dataset_leftbottom_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)       
        dataloader_midbottom_lr = th.utils.data.DataLoader(dataset_midbottom_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader_rightbottom_lr = th.utils.data.DataLoader(dataset_rightbottom_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)       
        dataloader_lr = th.utils.data.DataLoader(dataset_lr, batch_size=args.batch_size, shuffle=False, num_workers=16)  
        dataloader_left_lr = th.utils.data.DataLoader(dataset_left_lr, batch_size=args.batch_size, shuffle=False, num_workers=16) 
        if args.start_from_scratch:
            dataset = DummyDataset(len(dataset_lefttop_lr), rank=0, world_size=1)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader = zip(dataloader, dataloader_lefttop_lr, dataloader_midtop_lr, dataloader_righttop_lr, dataloader_leftbottom_lr, dataloader_midbottom_lr, dataloader_rightbottom_lr,dataloader_lr, dataloader_left_lr)

    # args.save_png_files=True
    if args.save_png_files:
        print(logger.get_dir())
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'images_left'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'mask'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'images_right'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'left_lr'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'right_lr'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'gt'), exist_ok=True)
        start_idx = args.global_rank * dataset.num_samples_per_rank

    logger.log("sampling...")
    all_images = []
    all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    for i, data in enumerate(dataloader):
        # if i<8:
        #     continue
        if args.use_img_for_guidance:
            image, label = data[0]
            image_lefttop_lr, label = data[1]
            image_lefttop_lr = image_lefttop_lr.to(device)
            image_midtop_lr, label = data[2]
            image_midtop_lr = image_midtop_lr.to(device)
            image_righttop_lr, label = data[3]
            image_righttop_lr = image_righttop_lr.to(device)
            image_leftbottom_lr, label = data[4]
            image_leftbottom_lr = image_leftbottom_lr.to(device)
            image_midbottom_lr, label = data[5]
            image_midbottom_lr = image_midbottom_lr.to(device)
            image_rightbottom_lr, label = data[6]
            image_rightbottom_lr = image_rightbottom_lr.to(device)
            image_lr, label = data[7]
            image_lr = image_lr.to(device)
            image_left_lr, label = data[8]
            image_left_lr = image_left_lr.to(device)
            cond_fn_lefttop = lambda x,t,light_factor_left,light_variance_left,y : color_cond_fn(x, t, light_factor=light_factor_left, light_variance=light_variance_left, y=y, x_lr=image_lefttop_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_midtop = lambda x,t,light_factor_left,light_variance_left,y : color_cond_fn(x, t, light_factor=light_factor_left, light_variance=light_variance_left, y=y, x_lr=image_midtop_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_righttop = lambda x,t,light_factor_left,light_variance_left,y : color_cond_fn(x, t, light_factor=light_factor_left, light_variance=light_variance_left, y=y, x_lr=image_righttop_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_leftbottom = lambda x,t,light_factor_left,light_variance_left,y : color_cond_fn(x, t, light_factor=light_factor_left, light_variance=light_variance_left, y=y, x_lr=image_leftbottom_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_midbottom = lambda x,t,light_factor_left,light_variance_left,y : color_cond_fn(x, t, light_factor=light_factor_left, light_variance=light_variance_left, y=y, x_lr=image_midbottom_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_rightbottom = lambda x,t,light_factor_left,light_variance_left,y : color_cond_fn(x, t, light_factor=light_factor_left, light_variance=light_variance_left, y=y, x_lr=image_rightbottom_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_left = lambda x,t,light_factor_right,light_variance_right,y : color_cond_fn_left(x, t, light_factor=light_factor_right, light_variance=light_variance_right, y=y, x_lr=image_left_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion_right, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
            cond_fn_right = lambda x,t,light_factor_right,light_variance_right,y : color_cond_fn_right(x, t, light_factor=light_factor_right, light_variance=light_variance_right, y=y, x_lr=image_lr, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion_right, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)
        else:
            image, label = data
            cond_fn = lambda x,t,y : general_cond_fn(x, t, y=y, x_lr=None)
        if args.start_from_scratch:
            shape = (image.shape[0], 3, args.image_size, 384)
            shape_lefttop = (image.shape[0], 3, args.image_size, 256)
            shape_midtop = (image.shape[0], 3, args.image_size, 256)
            shape_righttop = (image.shape[0], 3, args.image_size, 256)
            shape_leftbottom = (image.shape[0], 3, args.image_size, 256)
            shape_midbottom = (image.shape[0], 3, args.image_size, 256)
            shape_rightbottom = (image.shape[0], 3, args.image_size, 256)
        else:
            shape = list(image.shape)
        if args.start_from_scratch and not args.use_img_for_guidance:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(shape[0],), device=device)
        else:
            classes = label.to(device).long()
        # print(classes)
        # light_factor =  th.abs(th.randn([1], device=device))
        light_factor_left =  th.randn([1], device=device)/100
        light_variance_left =  th.rand([256,256], device=device)/10000
        light_factor_right =  th.randn([1], device=device)/100
        light_variance_right =  th.rand([256,384], device=device)/10000
        image = image.to(device)
        model_kwargs = {}
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_fn_right = (
            diffusion_right.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        if args.start_from_scratch:
            sample, light_factor_return, light_variance_return = sample_fn_right(
                model_fn_right,
                shape,
                light_factor_right, 
                light_variance_right,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn_left=cond_fn_left,
                cond_fn_right=cond_fn_right,
                device=device
            )
            print(light_variance_return.shape)
            m = torch.nn.Upsample(scale_factor=400/256, mode='bilinear')
            light_variance_return = m(light_variance_return.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            sample_lefttop, light_factor_final1, light_variance_final1 = sample_fn(
                model_fn,
                shape_lefttop,
                light_factor_return,
                light_variance_return[0:256, 0:256],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_lefttop,
                device=device
            )
            sample_midtop, light_factor_final2, light_variance_final2 = sample_fn(
                model_fn,
                shape_midtop,
                light_factor_return,
                light_variance_return[0:256, 172:428],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_midtop,
                device=device
            )
            sample_righttop, light_factor_final3, light_variance_final3 = sample_fn(
                model_fn,
                shape_righttop,
                light_factor_return,
                light_variance_return[0:256, 344:600],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_righttop,
                device=device
            )
            sample_leftbottom, light_factor_final4, light_variance_final4 = sample_fn(
                model_fn,
                shape_leftbottom,
                light_factor_return,
                light_variance_return[144:400, 0:256],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_leftbottom,
                device=device
            )

            sample_midbottom, light_factor_final5, light_variance_final5 = sample_fn(
                model_fn,
                shape_leftbottom,
                light_factor_return,
                light_variance_return[144:400, 172:428],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_midbottom,
                device=device
            )

            sample_rightbottom, light_factor_final6, light_variance_final6 = sample_fn(
                model_fn,
                shape_rightbottom,
                light_factor_return,
                light_variance_return[144:400, 344:600],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_rightbottom,
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
        sample_lefttop = ((sample_lefttop + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_lefttop = sample_lefttop.permute(0, 2, 3, 1)
        sample_lefttop = sample_lefttop.contiguous()

        sample_midtop = ((sample_midtop + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_midtop = sample_midtop.permute(0, 2, 3, 1)
        sample_midtop = sample_midtop.contiguous()

        sample_righttop = ((sample_righttop + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_righttop = sample_righttop.permute(0, 2, 3, 1)
        sample_righttop = sample_righttop.contiguous()

        sample_leftbottom = ((sample_leftbottom + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_leftbottom = sample_leftbottom.permute(0, 2, 3, 1)
        sample_leftbottom = sample_leftbottom.contiguous()

        sample_midbottom = ((sample_midbottom + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_midbottom = sample_midbottom.permute(0, 2, 3, 1)
        sample_midbottom = sample_midbottom.contiguous()

        sample_rightbottom = ((sample_rightbottom + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_rightbottom = sample_rightbottom.permute(0, 2, 3, 1)
        sample_rightbottom = sample_rightbottom.contiguous()

        image_lefttop_lr = ((image_lefttop_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_lefttop_lr = image_lefttop_lr.permute(0, 2, 3, 1)
        image_lefttop_lr = image_lefttop_lr.contiguous()

        image_midtop_lr = ((image_midtop_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_midtop_lr = image_midtop_lr.permute(0, 2, 3, 1)
        image_midtop_lr = image_midtop_lr.contiguous()

        image_righttop_lr = ((image_righttop_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_righttop_lr = image_righttop_lr.permute(0, 2, 3, 1)
        image_righttop_lr = image_righttop_lr.contiguous()

        image_leftbottom_lr = ((image_leftbottom_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_leftbottom_lr = image_leftbottom_lr.permute(0, 2, 3, 1)
        image_leftbottom_lr = image_leftbottom_lr.contiguous()

        image_midbottom_lr = ((image_midbottom_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_midbottom_lr = image_midbottom_lr.permute(0, 2, 3, 1)
        image_midbottom_lr = image_midbottom_lr.contiguous()

        image_rightbottom_lr = ((image_rightbottom_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_rightbottom_lr = image_rightbottom_lr.permute(0, 2, 3, 1)
        image_rightbottom_lr = image_rightbottom_lr.contiguous()

        light_variance_final1 = (light_variance_final1*10000).clamp(0, 255).to(th.uint8)
        light_variance_final1 = light_variance_final1.unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1)
        light_variance_final1 = light_variance_final1.contiguous()

        light_variance_final2 = (light_variance_final2*10000).clamp(0, 255).to(th.uint8)
        light_variance_final2 = light_variance_final2.unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1)
        light_variance_final2 = light_variance_final2.contiguous()

        light_variance_final3 = (light_variance_final3*10000).clamp(0, 255).to(th.uint8)
        light_variance_final3 = light_variance_final3.unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1)
        light_variance_final3 = light_variance_final3.contiguous()

        light_variance_final4 = (light_variance_final4*10000).clamp(0, 255).to(th.uint8)
        light_variance_final4 = light_variance_final4.unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1)
        light_variance_final4 = light_variance_final4.contiguous()

        light_variance_final5 = (light_variance_final5*10000).clamp(0, 255).to(th.uint8)
        light_variance_final5 = light_variance_final5.unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1)
        light_variance_final5 = light_variance_final5.contiguous()

        light_variance_final6 = (light_variance_final6*10000).clamp(0, 255).to(th.uint8)
        light_variance_final6 = light_variance_final6.unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1)
        light_variance_final6 = light_variance_final6.contiguous()
        
        sample_lefttop = sample_lefttop.detach().cpu().numpy()
        sample_midtop = sample_midtop.detach().cpu().numpy()
        sample_righttop = sample_righttop.detach().cpu().numpy()
        sample_leftbottom = sample_leftbottom.detach().cpu().numpy()
        sample_midbottom = sample_midbottom.detach().cpu().numpy()
        sample_rightbottom = sample_rightbottom.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        image_lefttop_lr = image_lefttop_lr.detach().cpu().numpy()
        image_midtop_lr = image_midtop_lr.detach().cpu().numpy()
        image_righttop_lr = image_righttop_lr.detach().cpu().numpy()
        image_leftbottom_lr = image_leftbottom_lr.detach().cpu().numpy()
        image_midbottom_lr = image_midbottom_lr.detach().cpu().numpy()
        image_rightbottom_lr = image_rightbottom_lr.detach().cpu().numpy()


        light_variance_final1 = light_variance_final1.detach().cpu().numpy()
        light_variance_final2 = light_variance_final2.detach().cpu().numpy()
        light_variance_final3 = light_variance_final3.detach().cpu().numpy()
        light_variance_final4 = light_variance_final4.detach().cpu().numpy()
        light_variance_final5 = light_variance_final5.detach().cpu().numpy()
        light_variance_final6 = light_variance_final6.detach().cpu().numpy()
        # if args.save_png_files:
        #     print(sample_left.shape)
        #     save_images(sample_left, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'images_left'))
        #     save_images(sample_right, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'images_right'))
        # save_images(image_left_lr, label.long(), start_idx + len(all_images) * args.batch_size, save_dir=os.path.join(logger.get_dir(), 'left_lr'))
        # save_images(image_right_lr, label.long(), start_idx + len(all_images) * args.batch_size, save_dir=os.path.join(logger.get_dir(), 'right_lr'))

        #拼接
        topil = transforms.ToPILImage()
        sample_lefttop = topil(torch.tensor(sample_lefttop).permute(0,3,1,2).squeeze(0))
        sample_midtop = topil(torch.tensor(sample_midtop).permute(0,3,1,2).squeeze(0))
        sample_righttop = topil(torch.tensor(sample_righttop).permute(0,3,1,2).squeeze(0))
        sample_leftbottom = topil(torch.tensor(sample_leftbottom).permute(0,3,1,2).squeeze(0))
        sample_midbottom = topil(torch.tensor(sample_midbottom).permute(0,3,1,2).squeeze(0))
        sample_rightbottom = topil(torch.tensor(sample_rightbottom).permute(0,3,1,2).squeeze(0))
        sample = topil(torch.zeros(3,400,600))
        sample.paste(sample_lefttop, box=(0,0), mask = None)
        sample_midtop = sample_midtop.crop((84,0,172,256))
        sample.paste(sample_midtop, box=(256,0), mask = None)
        sample_righttop = sample_righttop.crop((0,0,256,256))
        sample.paste(sample_righttop, box=(344,0), mask = None)
        sample_leftbottom = sample_leftbottom.crop((0,112,256,256))
        sample.paste(sample_leftbottom, box=(0,256), mask = None)
        sample_midbottom = sample_midbottom.crop((84,112,172,256))
        sample.paste(sample_midbottom, box=(256,256), mask = None)
        sample_rightbottom = sample_rightbottom.crop((0,112,256,256))
        sample.paste(sample_rightbottom, box=(344,256), mask = None)
        totensor = transforms.ToTensor()
        sample = np.array((totensor(sample)*255).unsqueeze(0).permute(0,2,3,1).to(th.uint8))

        save_images(sample, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'images'))

        topil = transforms.ToPILImage()
        light_variance_final1 = topil(torch.tensor(light_variance_final1).permute(0,3,1,2).squeeze(0))
        light_variance_final2 = topil(torch.tensor(light_variance_final2).permute(0,3,1,2).squeeze(0))
        light_variance_final3 = topil(torch.tensor(light_variance_final3).permute(0,3,1,2).squeeze(0))
        light_variance_final4 = topil(torch.tensor(light_variance_final4).permute(0,3,1,2).squeeze(0))
        light_variance_final5 = topil(torch.tensor(light_variance_final5).permute(0,3,1,2).squeeze(0))
        light_variance_final6 = topil(torch.tensor(light_variance_final6).permute(0,3,1,2).squeeze(0))
        sample_mask = topil(torch.zeros(3,400,600))
        sample_mask.paste(light_variance_final1, box=(0,0), mask = None)
        light_variance_final2 = light_variance_final2.crop((84,0,172,256))
        sample_mask.paste(light_variance_final2, box=(256,0), mask = None)
        light_variance_final3 = light_variance_final3.crop((0,0,256,256))
        sample_mask.paste(light_variance_final3, box=(344,0), mask = None)
        light_variance_final4 = light_variance_final4.crop((0,112,256,256))
        sample_mask.paste(light_variance_final4, box=(0,256), mask = None)
        light_variance_final5 = light_variance_final5.crop((84,112,172,256))
        sample_mask.paste(light_variance_final5, box=(256,256), mask = None)
        light_variance_final6 = light_variance_final6.crop((0,112,256,256))
        sample_mask.paste(light_variance_final6, box=(344,256), mask = None)
        totensor = transforms.ToTensor()
        sample_mask = np.array((totensor(sample_mask)*255).unsqueeze(0).permute(0,2,3,1).to(th.uint8))

        save_images(sample_mask, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'mask'))

        all_images.append(sample)
        all_labels.append(classes)
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # if args.save_numpy_array:
    #     arr = np.concatenate(all_images, axis=0)
    #     # arr = arr[: args.num_samples]
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     # label_arr = label_arr[: args.num_samples]

    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_rank_{args.global_rank}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr, label_arr)

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
    parser.add_argument("--save_dir", default='/mnt/lustre/feiben/DDPM_Beat_GAN/generated_image_x0_enhancement_brightness_lol_disco_mask', type=str, help='the directory to save the generate images')
    parser.add_argument("--save_png_files", action='store_true', help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    # these two arguments are only valid when not start from scratch
    parser.add_argument("--denoise_steps", default=25, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path", default='/mnt/lustre/feiben/DDPM_Beat_GAN/evaluations/precomputed/biggan_deep_imagenet64.npz', type=str, help='path to the generated images. Could be an npz file or an image folder')
    
    parser.add_argument("--use_img_for_guidance", action='store_true', help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')
    parser.add_argument("--img_guidance_scale", default=1000, type=float, help='guidance scale')
    parser.add_argument("--base_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_rightcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_left_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_leftcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_lefttop_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_lefttopcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_midtop_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_midtopcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_righttop_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_righttopcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_leftbottom_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_leftbottomcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_midbottom_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_midbottomcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_rightbottom_samples", default='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/LOL_rightbottomcrop_resolution_256.npz', type=str, help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
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
