from imagenet_dataset import ImageFolderDataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import torch

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

deg = "color"
image_size = 256
channels = 3
device = 'cpu'
H_funcs = None
if deg[:2] == 'cs':
    compress_by = int(deg[2:])
    from functions.svd_replacement import WalshHadamardCS
    loaded = np.loadtxt("./inp_masks/mask.np")
    mask = torch.from_numpy(loaded).to(device)
    H_funcs = WalshHadamardCS(channels, image_size, compress_by, mask.long(), device)
elif deg[:3] == 'inp':
    from functions.svd_replacement import Inpainting
    if deg == 'inp_lolcat':
        loaded = np.load("inp_masks/lolcat_extra.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
    elif deg == 'inp_lorem':
        loaded = np.load("inp_masks/lorem3.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
    else:
        loaded = np.loadtxt("./inp_masks/mask.np")
        mask = torch.from_numpy(loaded).to(device)
        missing_r = mask[:image_size**2 // 4].to(device).long() * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    H_funcs = Inpainting(channels, image_size, missing, device)
elif deg == 'deno':
    from functions.svd_replacement import Denoising
    H_funcs = Denoising(channels, image_size, device)
elif deg[:10] == 'sr_bicubic':
    factor = int(deg[10:])
    from functions.svd_replacement import SRConv
    def bicubic_kernel(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
        elif 1 < abs(x) and abs(x) < 2:
            return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
        else:
            return 0
    k = np.zeros((factor * 4))
    for i in range(factor * 4):
        x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    kernel = torch.from_numpy(k).float().to(device)
    H_funcs = SRConv(kernel / kernel.sum(), \
                        channels, image_size, device, stride = factor)
elif deg == 'deblur_uni':
    from functions.svd_replacement import Deblurring
    H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(device), channels, image_size, device)
elif deg == 'deblur_gauss':
    from functions.svd_replacement import Deblurring
    sigma = 10
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
    kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
    H_funcs = Deblurring(kernel / kernel.sum(), channels, image_size, device)
elif deg == 'deblur_aniso':
    from functions.svd_replacement import Deblurring2D
    sigma = 20
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
    kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
    sigma = 1
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
    kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
    H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), channels, image_size, device)
elif deg[:2] == 'sr':
    blur_by = int(deg[2:])
    from functions.svd_replacement import SuperResolution
    H_funcs = SuperResolution(channels, image_size, blur_by, device)
elif deg == 'color':
    from functions.svd_replacement import Colorization
    H_funcs = Colorization(image_size, device)
else:
    print("ERROR: degradation type not supported")
    quit()

if __name__ == '__main__':
    import pdb
    path='./imagenet/'
    image_size1 = 64
    image_size2 = 256

    transform1 = transforms.Compose([transforms.Resize(image_size1), # short side is image_size, short / long ratio is kept
            transforms.CenterCrop(image_size1)])
    dataset = ImageFolderDataset(path, permute=True, normalize=False, transform=transform1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=1)

    transform2 = transforms.Compose([transforms.Resize(image_size2), # short side is image_size, short / long ratio is kept
            transforms.CenterCrop(image_size2)])
    dataset2 = ImageFolderDataset(path, permute=True, normalize=False, transform=transform2)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=50, shuffle=False, num_workers=1)

    imgs = []
    imgs_high = []
    labels = []
    for i, (data, data2) in enumerate(zip(dataloader, dataloader2)):
        x, y = data
        x2, y2 = data2

        x2 = H_funcs.H(x2.to(torch.float32))
        x2_res = H_funcs.H_pinv(x2).view(x2.shape[0], 3, 256, 256)

        if deg[:6] == 'deblur': x2_res = x2.view(x2.shape[0], channels, image_size, image_size)
        elif deg == 'color': x2_res = x2.view(x2.shape[0], 1, image_size, image_size).repeat(1, 3, 1, 1)

        imgs.append(x)
        imgs_high.append(x2_res.to(torch.uint8))
        labels.append(y)

    imgs = torch.cat(imgs, dim=0)
    imgs_high = torch.cat(imgs_high, dim=0)
    labels = torch.cat(labels, dim=0)

    # file_name = deg + ('_resolution_%d' % image_size1)
    file_name_high =  deg + ('_resolution_%d' % image_size2)
    # save_image_tensor(imgs, labels, file_name)
    save_image_tensor(imgs_high, labels, file_name_high)

    imgs = imgs.permute(0,2,3,1) # shape BHWC
    imgs = imgs.detach().cpu().numpy()

    imgs_high = imgs_high.permute(0,2,3,1) # shape BHWC
    imgs_high = imgs_high.detach().cpu().numpy()

    labels = labels.detach().cpu().numpy()

    # file_name_1 = deg + ('_resolution_%d.npz' % image_size1)
    file_name_high_1 =   deg + ('_resolution_%d.npz' % image_size2)
    # np.savez(file_name_1, imgs, labels)
    np.savez(file_name_high_1, imgs_high, labels)
    # pdb.set_trace()