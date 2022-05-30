import glob
import os
import random

import torch
import numpy as np

import nibabel as nib

import numpy.ma as ma


if __name__ == '__main__':
    dummy_input = np.reshape(np.arange(2*2*3), (2, 2, 3))
    print('dummy volume:')
    print(dummy_input)
    print(np.shape(dummy_input))
    print('\n')

    # # Try with 2D:
    # dummy_slice = dummy_input[:, :, 0]
    # print('dummy slice:')
    # print(dummy_slice)
    # print(np.shape(dummy_slice))
    # print('\n')
    #
    # dummy_reflection = np.pad(dummy_slice, pad_width=(2, 2), mode='symmetric')
    # print('dummy reflection:')
    # print(dummy_reflection)
    # print(np.shape(dummy_reflection))
    # print('\n')

    # Try with 3D:
    dummy_volume = dummy_input[:, :, 0:2]
    print('dummy sub volume:')
    print(dummy_volume)
    print(np.shape(dummy_volume))
    print('\n')

    dummy_volume_reflection = np.pad(dummy_volume, pad_width=((2, 2), (3, 3), (4, 4)), mode='symmetric')
    print('dummy sub volume reflection:')
    print(dummy_volume_reflection)
    print(np.shape(dummy_volume_reflection))
    print('\n')
