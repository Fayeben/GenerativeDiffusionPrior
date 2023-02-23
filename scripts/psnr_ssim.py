import cv2
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# from lpips_pytorch import lpips
import lpips

import torch
import numpy as np
# define as a criterion module (recommended)
# criterion = LPIPS(
#     net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
#     version='0.1'  # Currently, v0.1 is supported
# )

import torchvision.transforms as transforms

sr_path = '/mnt/lustre/feiben/DDPM_Beat_GAN/HDR-DUIBi/naive'
hr_path = '/mnt/lustre/feiben/DDPM_Beat_GAN/HDR-DUIBi/GT-1'
degraded_path = '/mnt/lustre/feiben/DDPM_Beat_GAN/generated_image_color_withvariance/lr'

sr_dir = os.listdir(sr_path)
hr_dir = os.listdir(hr_path)
degraded_dir = os.listdir(degraded_path)

psnr = 0.0
ssim = 0.0
n = 0
lpips_l = 0.0
consistency_loss = 0.0
loss_fn = lpips.LPIPS(net='alex',version='0.1')
def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

deg = "light"
for hr_image in hr_dir:
    for sr_image in sr_dir:
        if sr_image == hr_image:
            if (sr_image[-3:]) != 'png':
                continue
            # print(sr_image, hr_image, sr_dir, hr_dir)
            # test = cv2.imread(os.path.join(sr_path, sr_image))
            # print(test.shape)
            compute_psnr = peak_signal_noise_ratio(cv2.imread(os.path.join(sr_path, sr_image)), cv2.imread(os.path.join(hr_path, hr_image)),data_range=255)
            compute_ssim = structural_similarity(to_grey(cv2.imread(os.path.join(sr_path, sr_image))),
                                        to_grey(cv2.imread(os.path.join(hr_path, hr_image))),data_range=255)
            # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
            psnr += compute_psnr
            ssim += compute_ssim

            img0 = lpips.im2tensor(lpips.load_image(os.path.join(sr_path, sr_image))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(hr_path, hr_image)))
            lpips_loss = loss_fn.forward(img0,img1)
            lpips_l += lpips_loss
            n += 1
            if n%100 == 0:
                print("finish compute [%d/%d]" % (n, len(hr_dir)))

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

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
        x = torch.from_numpy(x).permute(2,0,1).to(torch.float32).unsqueeze(0)
        # print(x.shape)
        x1 = x[:,0,:,:].unsqueeze_(1)
        x2 = x[:,1,:,:].unsqueeze_(1)
        x3 = x[:,2,:,:].unsqueeze_(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight)
        x2 = F.conv2d(x2, self.weight)
        x3 = F.conv2d(x3, self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        x = np.array(x.squeeze(0).permute(1,2,0))
        return x
    
    
def get_gaussian_blur(kernel_size, device):
    kernel = gkern(kernel_size, 2).astype(np.float32)
    gaussian_blur = GaussianBlur(kernel)
    return gaussian_blur.to(device)


if deg == "gray":
    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue
                compute_consistency =  mean_squared_error(to_grey(cv2.imread(os.path.join(sr_path, sr_image))),
                                            to_grey(cv2.imread(os.path.join(degraded_path, degraded_image))))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency
                
elif deg == "deblur":

    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue

                blur = get_gaussian_blur(kernel_size=9, device='cpu')
                # print(cv2.imread(os.path.join(sr_path, sr_image)).shape)
                # print((blur(cv2.imread(os.path.join(hr_path, hr_image)))).shape)
                compute_consistency =  mean_squared_error(blur(cv2.imread(os.path.join(sr_path, sr_image))),
                                            blur(cv2.imread(os.path.join(degraded_path, degraded_image))))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency

elif deg == "deblur_ddrm":
    image_size = 256
    channels = 3
    from functions.svd_replacement import Deblurring
    sigma = 10
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
    kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to('cpu')
    H_funcs = Deblurring(kernel / kernel.sum(), channels, image_size, 'cpu')
    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue
                # print(cv2.imread(os.path.join(sr_path, sr_image)).shape)
                # print((blur(cv2.imread(os.path.join(hr_path, hr_image)))).shape)
                # print(cv2.imread(os.path.join(sr_path, sr_image)))
                x_sr = H_funcs.H(torch.tensor(cv2.imread(os.path.join(sr_path, sr_image))).to(torch.float32).permute(2,0,1).unsqueeze(0))
                x_sr = np.array(x_sr.view(x_sr.shape[0], 3, 256, 256).squeeze(0).permute(1,2,0))
                compute_consistency =  mean_squared_error(x_sr,
                                            cv2.imread(os.path.join(degraded_path, degraded_image)))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency


elif deg == "inp":
    image_size = 256
    channels = 3
    from functions.svd_replacement import Inpainting
    loaded = np.loadtxt("/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/inp_masks/mask.np")
    mask = torch.from_numpy(loaded)
    missing_r = mask[:image_size**2 // 4].long() * 3  
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    H_funcs = Inpainting(channels, image_size, missing, device='cpu')
    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue
                # print(cv2.imread(os.path.join(sr_path, sr_image)).shape)
                # print((blur(cv2.imread(os.path.join(hr_path, hr_image)))).shape)
                # print(cv2.imread(os.path.join(sr_path, sr_image)))
                x_sr = H_funcs.H(torch.tensor(cv2.imread(os.path.join(sr_path, sr_image))).permute(2,0,1).unsqueeze(0))
                x_sr = np.array(H_funcs.H_pinv(x_sr).view(x_sr.shape[0], 3, 256, 256).squeeze(0).permute(1,2,0))
                compute_consistency =  mean_squared_error(x_sr,
                                            cv2.imread(os.path.join(degraded_path, degraded_image)))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency

elif deg == "inp_lorem":
    image_size = 256
    channels = 3
    from functions.svd_replacement import Inpainting

    loaded = np.load("inp_masks/lorem3.npy")
    mask = torch.from_numpy(loaded).to('cpu').reshape(-1)
    missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    H_funcs = Inpainting(channels, image_size, missing, device='cpu')
    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue
                # print(cv2.imread(os.path.join(sr_path, sr_image)).shape)
                # print((blur(cv2.imread(os.path.join(hr_path, hr_image)))).shape)
                # print(cv2.imread(os.path.join(sr_path, sr_image)))
                x_sr = H_funcs.H(torch.tensor(cv2.imread(os.path.join(sr_path, sr_image))).permute(2,0,1).unsqueeze(0).to(torch.float32))
                x_sr = np.array(H_funcs.H_pinv(x_sr).view(x_sr.shape[0], 3, 256, 256).squeeze(0).permute(1,2,0))
                compute_consistency =  mean_squared_error(x_sr,
                                            cv2.imread(os.path.join(degraded_path, degraded_image)))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency

elif deg == "inp_lolcat":
    image_size = 256
    channels = 3
    from functions.svd_replacement import Inpainting

    loaded = np.load("inp_masks/lolcat_extra.npy")
    mask = torch.from_numpy(loaded).to('cpu').reshape(-1)
    missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    H_funcs = Inpainting(channels, image_size, missing, device='cpu')
    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue
                # print(cv2.imread(os.path.join(sr_path, sr_image)).shape)
                # print((blur(cv2.imread(os.path.join(hr_path, hr_image)))).shape)
                # print(cv2.imread(os.path.join(sr_path, sr_image)))
                x_sr = H_funcs.H(torch.tensor(cv2.imread(os.path.join(sr_path, sr_image))).permute(2,0,1).unsqueeze(0).to(torch.float32))
                x_sr = np.array(H_funcs.H_pinv(x_sr).view(x_sr.shape[0], 3, 256, 256).squeeze(0).permute(1,2,0))
                print(x_sr.shape)
                print(cv2.imread(os.path.join(degraded_path, degraded_image)).shape)
                compute_consistency =  mean_squared_error(x_sr,
                                            cv2.imread(os.path.join(degraded_path, degraded_image)))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency

elif deg == "sr4":
    image_size = 256
    channels = 3
    blur_by = int(deg[2:])
    from functions.svd_replacement import SuperResolution
    H_funcs = SuperResolution(channels, image_size, blur_by, device='cpu')
    for degraded_image in degraded_dir:
        for sr_image in sr_dir:
            if sr_image == degraded_image:
                if (sr_image[-3:]) != 'png':
                    continue
                # print(cv2.imread(os.path.join(sr_path, sr_image)).shape)
                # print((blur(cv2.imread(os.path.join(hr_path, hr_image)))).shape)
                # print(cv2.imread(os.path.join(sr_path, sr_image)))
                x_sr = H_funcs.H(torch.tensor(cv2.imread(os.path.join(sr_path, sr_image))).permute(2,0,1).unsqueeze(0).to(torch.float32))
                x_sr = np.array(H_funcs.H_pinv(x_sr).view(x_sr.shape[0], 3, 256, 256).squeeze(0).permute(1,2,0))

                compute_consistency =  mean_squared_error(x_sr,
                                            cv2.imread(os.path.join(degraded_path, degraded_image)))
                # lpips_loss = lpips(cv2.imread(os.path.join(sr_dir, sr_image)), cv2.imread(os.path.join(hr_dir, hr_image)), net_type='alex', version='0.1')
                consistency_loss += compute_consistency

elif deg == "light":

    consistency_loss = 0

psnr = psnr / n
ssim = ssim / n
consistency_loss = consistency_loss/n
lpips_l = lpips_l / n
# lpips = lpips / n
print("average psnr = ", psnr)
print("average ssim = ", ssim)
print("average consistency = ", consistency_loss)
print("average lpips = ", lpips_l)

