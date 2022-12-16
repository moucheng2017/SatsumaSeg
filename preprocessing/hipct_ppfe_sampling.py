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
    all_slices.sort()
    all_slices = [os.path.join(img_path, a) for a in all_slices]
    starting = 450
    edges = 200

    for i, s in enumerate(all_slices):
        diff = i - starting
        if diff == 0:
            img = Image.open(s)
            img = np.asfarray(img)
            img = np.expand_dims(img, axis=0)
            img_volume = img
            print('slice' + s + ' done')
        elif 0 < diff:
            img = Image.open(s)
            img = np.asfarray(img)
            img = np.expand_dims(img, axis=0)
            img_volume = np.concatenate((img_volume, img), axis=0)
            print('slice' + s + ' done')
        else:
            pass

    img_volume = img_volume[:, edges:-edges, edges:-edges]
    img_volume = img_volume / 255.
    img_save_name = '/home/moucheng/projects_data/PPFE_HipCT/processed/imgs' + str() + '.npy'
    np.save(img_save_name, img_volume)
    print('slices saved from the tiff images.')

    # # np.save('')
    # slice_index = 100
    # plt.imshow(img_volume[slice_index, :, :])
    # plt.show()

