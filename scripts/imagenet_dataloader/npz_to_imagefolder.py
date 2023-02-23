from tqdm import tqdm
import numpy as np
import os
import shutil
import argparse

from PIL import Image

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

import pdb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../exps/imagenet/ddpm_250_steps/diffusion_and_reverse_25_steps_refine_sr_generated_samples/samples_all.npz', help='the npz dataset file')
    args = parser.parse_args()
    file_name = args.dataset
    path, name = os.path.split(file_name)

    data = np.load(file_name)
    imgs = data['arr_0']
    data.close()
    print('The dataset is of shape', imgs.shape)

    save_folder = os.path.join(path, 'images')
    os.makedirs(save_folder, exist_ok=True)
    for i in tqdm(range(imgs.shape[0])):
        x = imgs[i]
        save_name = 'ILSVRC2012_val_%s.png' % str(i+1).zfill(8)
        Image.fromarray(x).save(os.path.join(save_folder, save_name))

    files = os.listdir(save_folder)
    print('There are %d files in %s' % (len(files), save_folder))


    
            