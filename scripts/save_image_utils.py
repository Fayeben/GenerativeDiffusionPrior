import os
import numpy as np
from PIL import Image

def save_images(image, label, start_index, save_dir, max_index=np.inf):
    # image is a numpy array of shape NHWC
    # label is a numpy array of shape N
    for i in range(image.shape[0]):
        index = start_index + i
        save_single_image(image[i], label[i], index, save_dir, max_index=max_index)

def save_single_image(image, label, index, save_dir, max_index=np.inf):
    # image is a numpy array of shape HWC
    # image = normalize_and_quantize(image)
    if index < max_index:
        save_name = '%d_label_%d.png' % (index, label) 
        Image.fromarray(image).save(os.path.join(save_dir, save_name))

