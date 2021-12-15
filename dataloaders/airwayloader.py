import glob
import os
import torch
# import gzip
# import shutil
# import random
import errno
# # import pydicom
import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# from nipype.interfaces.ants import N4BiasFieldCorrection
from tifffile import imsave

# todo:
# Read cases, for each case, read the whole volume
# 1. random crop around the lung mask
# 2. crop again around the lung mask but with a lower resolution
# 3. apply contrast augmentation and noises to create another input


def foreground_mask(x, y, d1, d2):
    # x: image volume
    # y: label volume
    # two steps of cropping
    # firstly crop only z-dimension
    # x = x[d1:d2, :, :]
    # secondly crop the part containing the foreground
    # y = y[d1:d2, :, :]
    
    return x, y

# def volume_cropping()

# def mirror_padding()

# def random_contrast()

# def random_noises()


# if __name__ == '__main__':
#     Crop()
