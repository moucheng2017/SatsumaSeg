import os

import matplotlib.pyplot as plt
import numpy as np

from libs.Augmentations import *
from PIL import Image

if __name__ == '__main__':

    # check the image intensities:
    # img_path = '/home/moucheng/projects_data/GLE689_mismatch_data/GLE_689_substack_im/2.25um_GLE-698_pag-0.05_20000.tif'
    # img = Image.open(img_path)
    # img = np.asfarray(img)
    # img = img / 255.
    # plt.imshow(img)
    # plt.show()
    ## conclusions:
    # the background intensity is 135
    # the empty areas intensities are 0.0

    img_path = '/home/moucheng/projects_data/PPFE_HipCT/GLE_689_substack_im'
    all_slices = os.listdir(img_path)
    all_slices = [os.path.join(img_path, a) for a in all_slices]
    starting = 50
    edges = 445

    for i, a in enumerate(all_slices):

        img = Image.open(a)
        img = np.asfarray(img)
        img = np.expand_dims(img, axis=0)
        diff = i - starting

        if diff == 0:
            img_volume = img
            print('slice %d is done...', i)
        elif diff > 0:
            img_volume = np.concatenate((img_volume, img), axis=0)
            print('slice %d is done...', i)
        else:
            pass

    img_volume = img_volume[:, edges:, edges:]
    img_volume = img_volume / 255.
    np.save('/home/moucheng/projects_data/PPFE_HipCT/processed/imgs.npy', img_volume)
    print('slices saved from the tiff images.')

    # # np.save('')
    # slice_index = 100
    # plt.imshow(img_volume[slice_index, :, :])
    # plt.show()

