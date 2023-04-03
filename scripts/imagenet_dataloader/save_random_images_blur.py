from imagenet_dataset import ImageFolderDataset
from PIL import Image,ImageFilter
import torchvision.transforms as transforms
import numpy as np
import os
import torch
import random
import math
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
        print('kernel size is {0}.'.format(self.kernel_size))
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

def save_image_tensor(x, label, save_dir):
    # x is a tensor of shape BCHW
    # label is a tensor of shape B
    os.makedirs(save_dir, exist_ok=True)
    x = x.permute(0,2,3,1) # shape BHWC
    x = x.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    for i in range(label.shape[0]):
        save_name = str(i).zfill(3) + '_label_' + str(label[i]).zfill(4) + '.png'
        Image.fromarray(x[i]).save(os.path.join(save_dir, save_name))

if __name__ == '__main__':
    import pdb
    path='/mnt/lustre/feiben/DDPM_Beat_GAN/scripts/imagenet_dataloader/imagenet/'
    image_size1 = 64
    image_size2 = 256


    transform1 = transforms.Compose([transforms.Resize(image_size1), # short side is image_size, short / long ratio is kept
            transforms.CenterCrop(image_size1)])
    dataset = ImageFolderDataset(path, permute=True, normalize=False, transform=transform1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    transform2 = transforms.Compose([transforms.Resize(image_size2), # short side is image_size, short / long ratio is kept
            transforms.CenterCrop(image_size2)])
    dataset2 = ImageFolderDataset(path, permute=True, normalize=False, transform=transform2)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=1)
    blur = get_gaussian_blur(kernel_size=9, device='cpu')
    
    imgs = []
    imgs_high = []
    labels = []
    for i, (data, data2) in enumerate(zip(dataloader, dataloader2)):
        # if i>4:
        #     break
        x, y = data
        x2, y2 = data2
        x = blur(x.to(torch.float32))
        x2 = blur(x2.to(torch.float32))
        imgs.append(x.to(torch.uint8))
        imgs_high.append(x2.to(torch.uint8))
        labels.append(y)

    imgs = torch.cat(imgs, dim=0)
    imgs_high = torch.cat(imgs_high, dim=0)
    labels = torch.cat(labels, dim=0)

    # save_image_tensor(imgs, labels, 'blur_church_resolution_%d' % image_size1)
    save_image_tensor(imgs_high, labels, 'blur_resolution_%d' % image_size2)

    imgs = imgs.permute(0,2,3,1) # shape BHWC
    imgs = imgs.detach().cpu().numpy()
    imgs_high = imgs_high.permute(0,2,3,1) # shape BHWC
    imgs_high = imgs_high.detach().cpu().numpy()

    labels = labels.detach().cpu().numpy()
    np.savez('blur_resolution_%d.npz' % image_size2, imgs_high, labels)
    # np.savez('blur_church_resolution_%d.npz' % image_size1, imgs, labels)
    # pdb.set_trace()