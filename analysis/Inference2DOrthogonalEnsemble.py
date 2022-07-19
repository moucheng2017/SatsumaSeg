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
                      lung,
                      temperature=2
                      ):
    seg0 = seg_one_plaine(volume, model, 0, temperature)
    seg1 = seg_one_plaine(volume, model, 1, temperature)
    seg2 = seg_one_plaine(volume, model, 2, temperature)

    seg = (seg0 + seg1 + seg2) / 3

    del seg0
    del seg1
    del seg2

    lung = np.transpose(lung.squeeze(), (1, 2, 0))
    seg = seg*lung
    del lung

    return seg


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


def ensemble(seg_path,
             threshold):
    final_seg = None
    all_segs = os.listdir(seg_path)
    all_segs.sort()
    all_segs = [os.path.join(seg_path, seg_name) for seg_name in all_segs if 'prob' in seg_name]
    for seg in all_segs:
        seg_ = np.load(seg)
        final_seg += seg_
        del seg_

    output = final_seg / len(all_segs)
    output = np.where(output > threshold, 1, 0)
    return output


def main(test_data_path,
         case,
         lung_path,
         model_path,
         temp,
         threshold=0.8,
         step_range=20000):

    # generate save path:
    save_path = os.path.abspath(model_path)
    save_path = os.path.join(save_path, 'segmentation')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # sort out all models:
    all_models = os.listdir(model_path)
    all_models.sort()

    # ran inference for each model:
    for model_name in all_models:
        step = model_name.split('_')[-1]
        step = float(step.split('.')[0])
        if step >= step_range:
            model_name = os.path.join(model_path, model_name)

            # nii --> np:
            data = nii2np(test_data_path)
            lung = nii2np(lung_path)

            # load model:
            model = torch.load(model_name)
            model.eval()

            # np --> tensor
            data = np2tensor(data)

            # segmentation 3 orthogonal planes:
            seg = seg_three_plaines(data, model, lung, temp)

            # save prepration:
            save_name = case + '_s' + str(step) + '_prob.nii.gz'

            # save seg:
            save_seg(save_path,
                     save_name,
                     test_data_path,
                     seg)

    # ensemble all segmentation files:
    final_seg = ensemble(save_path, threshold)
    save_name = case + '_final_prob.nii.gz'
    # save seg:
    save_seg(save_path,
             save_name,
             test_data_path,
             final_seg)

    print('Done')


if __name__ == "__main__":

    case = '6357B'
    threshold = 0.5
    sliding_window = 1
    temperature = 2

    # data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/imgs/' + case + '.nii.gz'
    # lung_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/lung/' + case + '_lunglabel.nii.gz'
    # model_path = '/home/moucheng/projects_codes/Results/airway/2022_07_04/OrthogonalSup2DSingle_e1_l0.0001_b4_w24_s50000_r0.001_c_False_n_False_t1.0/trained_models/'

    data_path = '/SAN/medic/PerceptronHead/data/lung/private/airway/test/imgs/' + case + '.nii.gz'
    lung_path = '/SAN/medic/PerceptronHead/data/lung/private/airway/test/lung/' + case + '_lunglabel.nii.gz'
    model_path = '/SAN/medic/PerceptronHead/Results/OrthogonalSup2DSingle_e1_l0.0001_b4_w24_s50000_r0.001_c_False_n_False_t1.0/trained_models/'

    main(data_path,
         case,
         lung_path,
         model_path,
         temperature,
         threshold,
         step_range=20000)




