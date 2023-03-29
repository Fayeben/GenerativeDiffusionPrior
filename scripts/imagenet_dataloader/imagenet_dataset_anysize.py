import os
from PIL import Image
import numpy as np
import pickle

import torchvision.transforms as transforms
import torch
import torch.utils.data as data


class ImageFolderDataset(data.Dataset):
    def __init__(self, folder_path, label_file='imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=0, world_size=1, return_numpy=False):
        self.folder_path = folder_path
        self.imgs = []
        valid_images = ['.jpeg', '.png', '.jpg']
        for f in os.listdir(self.folder_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_images:
                self.imgs.append(f)
        
        if 'label' in self.imgs[0]:
            # img names is like 27987_label_762.png
            self.imgs = sorted(self.imgs, key = lambda x : int(x.split('_')[0]))
        else:
            self.imgs = sorted(self.imgs)
        print('Find %d images in the folder %s' % (len(self.imgs), self.folder_path))

        if world_size > 1:
            num_samples_per_rank = int(np.ceil(len(self.imgs) / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.imgs = self.imgs[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = len(self.imgs)

        self.len = len(self.imgs)
        print('This process hanldes %d images in the folder %s' % (self.len, self.folder_path))
        
        
        self.transform = transform # this transform should only perform PIL image tranforms
        # transforms to tensors like normalizations should be performed by other params
        self.permute = permute
        self.normalize = normalize
        self.return_numpy = return_numpy

        # if label_file == 0, return dummy label 0
        # elif label_file is a pickle file, use labels in stored in the label_file 
        # else label_file is None, label in image name, return label in the image name
        if not label_file is None:
            if isinstance(label_file, int):
                self.labels = label_file
            else:
                handle = open(label_file, 'rb')
                data = pickle.load(handle)
                handle.close()
                self.labels = data['imgs_label_dict']
        else:
            self.labels = None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = self.imgs[index]
        x = Image.open(os.path.join(self.folder_path, img_name))
        if not self.transform is None:
            x = self.transform(x)

        x = torch.from_numpy(np.array(x))
        # img.close()
        if len(x.shape) == 2:
            x = torch.stack([x,x,x], dim=2)
        if x.shape[2] == 4:
            x = x[:,:,0:3]
        
        if self.permute:
            x = x.permute(2,0,1)
        if self.normalize:
            # before normalize, x is of dtype uint8 range from 0 to 255
            # after normalize, we make x float type and range from -1 to 1
            x = x.to(torch.float)
            x = x/255 * 2 - 1

        if self.return_numpy:
            x = x.detach().numpy()

        if isinstance(self.labels, int):
            y = self.labels
        elif isinstance(self.labels, dict):
            # img_name is like ILSVRC2012_val_00034973.JPEG
            # key is like 00034973
            key = img_name.split('.')[0]
        #     y = self.labels[key]
        # else:
        #     # img names is like 27987_label_762.png
        #     y = int(img_name.split('_')[-1].split('.')[0])
        y = 990
        return x, y


if __name__ == '__main__':
    import pdb
    path='/data/feiben1/fb'

    image_size = 256
    transform = transforms.Compose(
            [   
                transforms.Resize(image_size), # short side is image_size, short / long ratio is kept
                transforms.CenterCrop(image_size)
            ]
        )

    dataset = ImageFolderDataset(path, transform=None, permute=True, normalize=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # for i in range(len(dataset)):
    #     x = dataset.__getitem__(i)
    #     print(x.shape)
    
    for i, data in enumerate(dataloader):
        x, y = data
        print(x.shape)
        pdb.set_trace()
