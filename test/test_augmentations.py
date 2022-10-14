import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np

import torch
import random

import numpy as np
import scipy.ndimage

sys.path.append('..')
from libs.Augmentations import *

if __name__ == "__main__":
    # read the file
    img_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/3d_binary/R176/C3/D3_S3/N2/labelled/patches'
    lbl_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/3d_binary/R176/C3/D3_S3/N2/labelled/labels'

    all_imgs = os.listdir(img_folder)
    all_imgs = [os.path.join(img_folder, img) for img in all_imgs]
    all_imgs.sort()

    all_lbls = os.listdir(lbl_folder)
    all_lbls = [os.path.join(lbl_folder, lbl) for lbl in all_lbls]
    all_lbls.sort()

    img_no = 1

    img = all_imgs[img_no]
    img = np.load(img)
    img = img[1, :, :]

    lbl = all_lbls[img_no]
    lbl = np.load(lbl)
    lbl = lbl[1, :, :]

    # random zoom in augmentation:
    zoom = RandomZoom(5)
    img_zoomed, lbl_zoomed = zoom.forward(img, lbl)

    # plot
    fig, axs = plt.subplot(2, 2)
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(img_zoomed)
    axs[1, 0].imshow(lbl)
    axs[1, 1].imshow(lbl_zoomed)
    plt.show()
