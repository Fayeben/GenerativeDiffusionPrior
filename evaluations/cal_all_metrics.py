import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2
from PIL import Image
import argparse

import torch
import lpips
import torch.nn.functional as F
from pytorch_ignite_psnr_ssim import compute_psnr_and_ssim

import sys
sys.path.append('../')
from scripts.imagenet_dataloader.imagenet_dataset import ImageFolderDataset
from scripts.npz_dataset import NpzDataset

import pdb

def get_dataset(path):
    if os.path.isfile(path): # samples could be store in a .npz file
        dataset = NpzDataset(path, normalize=False, permute=False)
    else:
        dataset = ImageFolderDataset(path, label_file='../scripts/imagenet_dataloader/imagenet_val_labels.pkl', transform=None, 
                        permute=False, normalize=False, return_numpy=True)
    return dataset


def compute_consistency(hr, lr, device):
    # hr, lr are images of shape HW3, hw3 respetively, uint8, range from 0 to 255
    lr_h, lr_w = lr.shape[0], lr.shape[1]
    hr = torch.from_numpy(hr).to(device)
    hr = hr.permute(2,0,1).unsqueeze(0) # 13HW
    hr = hr.float() / 255 # 0 to 1
    hr = F.interpolate(hr, size=(lr_h, lr_w), mode='bicubic', antialias=True) # 13hw
    hr = hr[0].permute(1,2,0) # hw3
    hr = torch.clamp(hr * 255, min=0, max=255) # 0 to 255
    hr = torch.round(hr)

    lr = torch.from_numpy(lr).to(device).float() # hw3
    mse = torch.mean((hr-lr)**2).cpu().item()
    return mse
    

def cv2torch(image, device):
    image = image[..., ::-1].astype(np.float32) #HW3
    image = torch.from_numpy(image).permute(2,0,1) #3HW
    image = image.to(device) / 255
    return image.unsqueeze(0)

def main(args):
    device = torch.device('cuda')
    all_metrics = {metric:[] for metric in args.metrics}
    # if 'fid' in args.metrics:
    #     # remind that fid&IS are both sensetive to the number of testing images
    #     assert args.ref_HR is not None
    #     fid_is = calculate_metrics(args.out_HR, args.ref_HR, cuda=True, isc=True, fid=True, verbose=False)
    #     all_metrics['fid'] = fid_is['frechet_inception_distance']
    #     all_metrics['is'] = fid_is['inception_score_mean']

    if 'lpips' in args.metrics:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # img_list = sorted(glob(osp.join(args.out_HR, '*.png')))
    dataset_out_HR = get_dataset(args.out_HR)
    dataset_gt_HR = get_dataset(args.GT_HR)
    dataset_input_LR = get_dataset(args.input_LR)
    
    for idx in tqdm(range(len(dataset_out_HR))):
        # pred_im = cv2.imread(img_file)
        pred_im = dataset_out_HR.__getitem__(idx)[0]
        # if 'label' in img_file:
        #     # img_file is like 5_label_7.png
        #     idx = int(img_file.split('_')[0])
        #     im_id = 'ILSVRC2012_val_%s' % str(idx+1).zfill(8)
        # else:
        #     im_id = osp.basename(img_file)[:23]
        # gt_im = cv2.imread(osp.join(args.GT_HR, f'{im_id}.png'))
        gt_im = dataset_gt_HR.__getitem__(idx)[0]
        if 'niqe' in args.metrics:
            all_metrics['niqe'].append(calculate_niqe(pred_im, crop_border=4, num_thread=8))
        
        psnr, ssim = compute_psnr_and_ssim(pred_im, gt_im, device, crop_border=4)
        all_metrics['psnr'].append(psnr)
        all_metrics['ssim'].append(ssim)

        if 'lpips' in args.metrics:
            with torch.no_grad():
                cur_lpips = loss_fn_vgg(cv2torch(pred_im, device=device), cv2torch(gt_im, device=device)).mean().cpu().item()
            all_metrics['lpips'].append(cur_lpips)
        if 'consistency' in args.metrics:
            # assert args.input_LR is not None
            # input_lr_im = cv2.imread(osp.join(args.input_LR, f'{im_id}.png')).astype(np.float64)
            input_lr_im = dataset_input_LR.__getitem__(idx)[0]
            # down_lr_im = generate_bicubic_img(pred_im).astype(np.float64)
            # mse = np.mean((input_lr_im - down_lr_im)**2)
            mse = compute_consistency(pred_im, input_lr_im, device)
            all_metrics['consistency'].append(mse)

    for k, v in all_metrics.items():
        print(k, np.mean(v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_LR', type=str, default='../datasets/imagenet/val_64.npz', help='The input LR image path')
    parser.add_argument('--GT_HR', type=str, default='../datasets/imagenet/val_256.npz', help='the GT HR image path')
    parser.add_argument('--out_HR', type=str, default='/home/zylyu/new_pool/ddpms/My_BasicSR/results/ESRGAN_SRx4_DF2KOST_official/visualization/ImageNet', help='the output/predicted SR image path')
    # parser.add_argument('--ref_HR', type=str, default='./train_subset/all_images', help='the reference HR image path')
    # parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'fid', 'is', 'lpips', 'niqe', 'consistency'])
    # parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'lpips', 'niqe', 'consistency'])
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'lpips', 'consistency'])
    parser.add_argument('--device', type=str, default='0', help='the cuda device to use for evaluation')
    args = parser.parse_args()

    if args.device == 'none':
        print('Using slurm allocated device:', os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    with torch.no_grad():
        main(args=args)
