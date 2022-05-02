import nibabel
import torch
import random
import numpy as np
import os

import glob
import tifffile as tiff

import errno
import imageio

from PIL import Image

import numpy.ma as ma

from dataloaders.Dataloader import RandomContrast
from collections import deque

# Work flow for each case:
# 1. Read nii.gz
# 1.5. Prepare an empty volume for storing slices
# 2. Transform nii.gz into np
# 3. Slice np file
# 4. Expand the dim of the slice for inference
# 5. Segmentation
# 6. Store them in NIFTI file format


def nii2np(file_path):
    # Read image:
    data = nibabel.load(file_path)
    data = data.get_fdata()
    data = np.array(data, dtype='flaot32')
    # Now applying lung window:
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    # H X W X D --> D X H X W:
    data = np.transpose(data, (2, 0, 1))
    if len(np.shape(data)) == 3:
        data = np.expand_dims(data, axis=0) # 1 X D X H X W
    return data


def np2tensor(data):
    data = torch.from_numpy(data).to(device='cuda', dtype=torch.float32)
    if len(data.size()) == 3:
        data = torch.unsqueeze(data, dim=0)
    return data


def np2tensor_batch(data_list):
    new_data_list = deque()
    for data in data_list:
        data = np2tensor(data)
        new_data_list.append(data)
    return new_data_list

def normalisation(lung, image):
    image_masked = ma.masked_where(lung > 0.5, image)
    lung_mean = np.nanmean(image_masked)
    lung_std = np.nanstd(image_masked)
    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


def adjustcontrast(data,
                   lung_mask,
                   adjust_times=0):
    outputs = deque()
    if adjust_times == 0:
        data = normalisation(lung=lung_mask, image=data)
        return outputs.append(data)
    else:
        contrast_augmentation = RandomContrast(bin_range=[100, 250])
        for i in range(adjust_times):
            data_ = contrast_augmentation.randomintensity(data)
            data_ = normalisation(lung=lung_mask, image=data_)
            outputs.append(data_)
        return outputs


def segment_single_slice(img,
                         model,
                         new_size):

    if len(np.shape(img)) == 2:
        img = np.expand_dims(img, axis=0) # 1 X 1 X H X W
        img = np.expand_dims(img, axis=0)
    elif len(np.shape(img)) == 3:
        img = np.expand_dims(img, axis=0)

    img = torch.from_numpy(img).to(device='cuda', dtype=torch.float32)
    b, d, h, w = torch.size(img)
    assert h == w >= new_size

    location_list = {
        "left_up": [0, 0],
        "center_up": [0, (w-new_size) // 2],
        "right_up": [0, w-new_size],
        "left_middle": [(h - new_size)//2, 0],
        "center_middle": [(h - new_size)//2, (w-new_size) // 2],
        "right_middle": [(h - new_size)//2, w-new_size],
        "left_bottom": [h-new_size, 0],
        "center_bottom": [h-new_size, (w-new_size) // 2],
        "right_bottom": [h-new_size, w-new_size]
    }

    for each_location, each_coordiate in location_list.items():
        cropped = img[:, :, each_coordiate[0]:each_coordiate[0]+new_size, each_coordiate[1]:each_coordiate[1]+new_size]
        cropped = torch.unsqueeze(cropped, dim=0)
        seg_slice = model(cropped)



    return seg_slice


def segment_single_case(test_data_path,
                        lung_path,
                        new_size):

    # nii --> np:
    data = nii2np(test_data_path)
    lung = nii2np(lung_path)

    # ensemble on random contrast augmented of image:
    augmented_data_list = adjustcontrast(data, lung, adjust_times=3)

    # np --> tensor:
    augmented_data_list = np2tensor(augmented_data_list)

    # prepare output:
    output = torch.zeros_like(augmented_data_list[0])

    # inference on each contrast:
    for augmented_data in augmented_data_list:
        temp_output = torch.zeros_like(augmented_data)




