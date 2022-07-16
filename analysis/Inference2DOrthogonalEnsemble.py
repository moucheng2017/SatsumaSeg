import nibabel
import torch
import random
import numpy as np
import os
import argparse
import glob
import tifffile as tiff

import errno
import imageio

from PIL import Image

import numpy.ma as ma

from dataloaders.Dataloader import RandomContrast
from collections import deque

import matplotlib.pyplot as plt


def nii2np(file_path):
    # Read image:
    data = nibabel.load(file_path)
    data = data.get_fdata()
    data = np.array(data, dtype='float32')
    print(np.shape(data))
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
    if len(data.size()) == 2:
        data = torch.unsqueeze(data, dim=0)
    return data


def np2tensor_batch(data_list):
    # print(len(data_list))
    # new_data_list = deque()
    new_data_list = []
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
    # outputs = deque()
    outputs = []
    if adjust_times == 0:
        data = normalisation(lung=lung_mask, image=data)
        return outputs.append(data)
    else:
        contrast_augmentation = RandomContrast(bin_range=[10, 250])
        for i in range(adjust_times):
            data_ = contrast_augmentation.randomintensity(data)
            data_ = normalisation(lung=lung_mask, image=data_)
            outputs.append(data_)
        return outputs


def seg_one_plaine(volume,
                   model,
                   direction=0,
                   temperature=2
                   ):

    c, d, h, w = volume.size()

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    if direction == 0:
        slice_no = d
    elif direction == 1:
        slice_no = h
    elif direction == 2:
        slice_no = w

    for i in range(0, slice_no, 1):

        if direction == 0:
            img = volume[:, i, :, :] # B x 1 x 512 x 512
            img = img.unsqueeze(1)
        elif direction == 1:
            img = volume[:, :, i, :] # B x 512 x 1 x 512
            img = img.unsqueeze(1)
        elif direction == 2:
            img = volume[:, :, :, i] # B x 512 x 512 x 1
            img = img.unsqueeze(1)

        seg_, _ = model(img)
        seg_ = torch.sigmoid(seg_ / temperature)
        seg_ = seg_.squeeze().detach().cpu().numpy() # W x D x H
        if direction == 0:
            seg[:, i, :, :] = np.expand_dims(seg_, axis=1)
        elif direction == 1:
            seg[:, :, i, :] = np.expand_dims(seg_, axis=2)
        elif direction == 2:
            seg[:, :, :, i] = np.expand_dims(seg_, axis=3)

    return seg


def seg_three_plaines(volume,
                      model,
                      temperature=2
                      ):
    seg0 = seg_one_plaine(volume, model, 0, temperature)
    seg1 = seg_one_plaine(volume, model, 1, temperature)
    seg2 = seg_one_plaine(volume, model, 2, temperature)

    seg = (seg0 + seg1 + seg2) / 3

    del seg0
    del seg1
    del seg2

    return seg


def merge_segs(folder):
    all_files = os.listdir(folder)
    seg = None
    for each_seg in all_files:
        each_seg = np.load(each_seg)
        seg += each_seg

    seg = seg / len(all_files)
    seg = (seg > 0.9).float()
    return seg


def segment2D(test_data_path,
              lung_path,
              model_path,
              threshold,
              new_size_d,
              new_size_h,
              new_size_w,
              sliding_window,
              temperature=2):

    d = [new_size_d, new_size_h, new_size_w]
    h = [new_size_h, new_size_d, new_size_w]
    w = [new_size_h, new_size_w, new_size_d]

    # nii --> np:
    data = nii2np(test_data_path)
    lung = nii2np(lung_path)

    # load model:
    model = torch.load(model_path)
    model.eval()

    data = np2tensor(data)
    output_w = seg_w_direction(data, model, w, sliding_window, temperature)
    output_h = seg_h_direction(data, model, h, sliding_window, temperature)
    output_d = seg_d_direction(data, model, d, sliding_window, temperature)
    output_prob = (output_d + output_h + output_w) / 3

    lung = np.transpose(lung.squeeze(), (1, 2, 0))
    output_prob = output_prob*lung
    output = np.where(output_prob > threshold, 1, 0)

    return np.squeeze(output), np.squeeze(output_prob)


def ensemble(seg_path):
    final_seg = None
    all_segs = os.listdir(seg_path)
    all_segs.sort()
    all_segs = [os.path.join(seg_path, seg_name) for seg_name in all_segs if 'prob' in seg_name]
    for seg in all_segs:
        seg = np.load(seg)
        final_seg += seg
    return final_seg / len(all_segs)


def save_seg(save_path,
             save_name,
             nii_path,
             saved_data):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    nii = nibabel.load(nii_path)
    segmentation_nii = nibabel.Nifti1Image(saved_data,
                                           nii.affine,
                                           nii.header)
    save_path_nii = os.path.join(save_path, save_name)
    nibabel.save(segmentation_nii, save_path_nii)


if __name__ == "__main__":
    case = '6357B'
    new_size_d = 5
    new_resolution_w = 480
    new_resolution_h = 320
    threshold = 0.5
    sliding_window = 1
    temperature = 2

    save_path = '/home/moucheng/PhD/2022_12_Clinical/orthogonal2d/preliminary/seg'
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/imgs/' + case + '.nii.gz'
    lung_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/lung/' + case + '_lunglabel.nii.gz'
    model_path = '/home/moucheng/PhD/2022_12_Clinical/orthogonal2d/preliminary/airway/2022_05_13/OrthogonalSup2D_e1_l0.001_b6_w64_s5000_r0.01_z5_t2.0/trained_models/'
    model_name = 'OrthogonalSup2D_e1_l0.001_b6_w64_s5000_r0.01_z5_t2.0_2000.pt'
    model_path_full = model_path + model_name
    save_name = case + '_seg2Dorthogonal_t' + str(threshold) + '_h' + str(new_resolution_h) + '_w' + str(new_resolution_w) + '_s' + str(sliding_window) + '_t' + str(temperature) + '.nii.gz'
    save_name_prob = case + '_prob2Dorthogonal_t' + str(threshold) + '_h' + str(new_resolution_h) + '_w' + str(new_resolution_w) + '_s' + str(sliding_window) + '_t' + str(temperature) + '.nii.gz'

    segmentation, probability = segment2D(data_path,
                                          lung_path,
                                          model_path_full,
                                          threshold,
                                          new_size_d,
                                          new_resolution_h,
                                          new_resolution_w,
                                          sliding_window,
                                          temperature)

    save_seg(save_path, save_name, data_path, segmentation)
    save_seg(save_path, save_name_prob, data_path, probability)

    print(np.shape(segmentation))
    print('End')


