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


# For consistency restrictions:
# 1. Global and local, same resolution
# 2. High resolution and low resolution


class GlobalLocal(object):
    def __init__(self, global_local_ratios=[0.5, 0.5, 0.5]):
        self.global_local_ratios = global_local_ratios

    def crop(self, global_input, global_seg):
        b, c, d, h, w = global_input.size()
        new_d = int(d * self.global_local_ratios[0])
        new_h = int(h * self.global_local_ratios[1])
        new_w = int(w * self.global_local_ratios[2])

        top_h = np.random.randint(0, h - new_h)
        top_w = np.random.randint(0, w - new_w)
        top_d = np.random.randint(0, d - new_d)

        local_input = global_input[
                      :,
                      :,
                      top_d:top_d + new_d,
                      top_h:top_h + new_h,
                      top_w:top_w + new_w
                      ]

        local_seg = global_seg[
                      :,
                      :,
                      top_d:top_d + new_d,
                      top_h:top_h + new_h,
                      top_w:top_w + new_w
                      ]

        return local_input, local_seg


class HighLow(object):
    def __init__(self, high_low_ratios=[2, 2, 2]):
        self.high_low_ratios = high_low_ratios

    def crop(self, global_input, global_seg):
        local_input = global_input[
                      :,
                      ::self.high_low_ratios[0],
                      ::self.high_low_ratios[1],
                      ::self.high_low_ratios[2]
                      ]

        local_seg = global_seg[
                      :,
                      ::self.high_low_ratios[0],
                      ::self.high_low_ratios[1],
                      ::self.high_low_ratios[2]
                      ]

        return local_input, local_seg

#
# if __name__ == '__main__':
#     dummy_input = np.random.rand(1, 512, 512, 480)


