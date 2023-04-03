from imagenet_dataset_anysize import ImageFolderDataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import torch

#!/usr/bin/env python

import getopt
import numpy
import PIL
import PIL.Image
import sys
import torch


def save_image_tensor(x, label, save_dir):
    # x is a tensor of shape BCHW
    # label is a tensor of shape B
    os.makedirs(save_dir, exist_ok=True)
    x = x.permute(0,2,3,1) # shape BHWC
    x = x.detach().cpu().numpy().astype(numpy.uint8)
    label = label.detach().cpu().numpy().astype(numpy.uint8)

    for i in range(label.shape[0]):
        # save_name = str(i).zfill(3) + '_label_' + str(label[i]).zfill(4) + '.png'
        save_name = str(i) + '_label_' + str('197') + '.png'
        Image.fromarray(x[i]).save(os.path.join(save_dir, save_name))

if __name__ == '__main__':
    import pdb
    path='/NTIRE-100/NTIRE-100-long'
    image_size1 = 64
    image_size2 = 256
    dataset = ImageFolderDataset(path, permute=True, normalize=False, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    dataset2 = ImageFolderDataset(path, permute=True, normalize=False, transform=None)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)

    imgs = []
    imgs_high = []
    labels = []
    for i, (data, data2) in enumerate(zip(dataloader, dataloader2)):

        x, y = data
        x2, y2 = data2
 
        imgs.append(x)
        imgs_high.append(x2)
        labels.append(y)

    imgs = torch.cat(imgs, dim=0)
    imgs_high = torch.cat(imgs_high, dim=0)
    labels = torch.cat(labels, dim=0)

    # file_name = ('hdr_nature_medium_resolution_%d' % image_size1)
    file_name_high = ('hdr_100_long_resolution_%d' % image_size2)
    # save_image_tensor(imgs, labels, file_name)
    save_image_tensor(imgs_high, labels, file_name_high)

    # imgs = imgs.permute(0,2,3,1) # shape BHWC
    # imgs = imgs.detach().cpu().numpy()

    imgs_high = imgs_high.permute(0,2,3,1) # shape BHWC
    imgs_high = imgs_high.detach().cpu().numpy().astype(numpy.uint8)

    labels = labels.detach().cpu().numpy().astype(numpy.uint8)

    # file_name_1 = ('hdr_valid_medium_resolution_%d.npz' % image_size1)
    file_name_high_1 = ('hdr_100_long_resolution_%d.npz' % image_size2)
    # np.savez(file_name_1, imgs, labels)
    np.savez(file_name_high_1, imgs_high, labels)
    # pdb.set_trace() 