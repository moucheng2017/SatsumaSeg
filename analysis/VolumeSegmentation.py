import torch
import sys
sys.path.append("..")
# torch.manual_seed(0)
import errno
import numpy as np
# import pandas as pd
import os
from os import listdir
# import Image

import timeit
import torch.nn as nn
import torch.nn.functional as F

import glob

import random
import matplotlib.pyplot as plt

import nibabel as nib


def segment_whole_volume(model,
                         volume,
                         train_size,
                         class_no=2,
                         full_resolution=False):
    '''
    volume: c x d x h x w
    model: loaded model
    calculate iou for each subvolume then sum them up then average, don't ensemble the volumes in gpu
    '''
    c, d, h, w = np.shape(volume)
    segmentation = np.zeros_like(volume)
    model.eval()
    # Loop through the whole volume:

    if full_resolution is False:
        for i in range(0, d - train_size[0]-1, train_size[0]//2):
            for j in range(0, h - train_size[1]-1, train_size[1]//2):
                for k in range(0, w - train_size[2]-1, train_size[2]//2):
                    subvolume = volume[:, i:i+train_size[0], j:j+train_size[1], k:k+train_size[2]]
                    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
                    subseg, _ = model(subvolume.unsqueeze(0))

                    if class_no == 2:
                        subseg = torch.sigmoid(subseg)
                        # subseg = (subseg > 0.5).float()
                    else:
                        subseg = torch.softmax(subseg, dim=1)
                        # _, subseg = torch.max(subseg, dim=1)

                    segmentation[:, i:i+train_size[0], j:j+train_size[1], k:k+train_size[2]] = subseg.detach().cpu().numpy()
    else:
        for i in range(0, d - train_size[0] - 1, train_size[0] // 2):
            subvolume = volume[:, i:i + train_size[0], :, :]
            subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
            subseg, _ = model(subvolume.unsqueeze(0))

            if class_no == 2:
                subseg = torch.sigmoid(subseg)
                # subseg = (subseg > 0.5).float()
            else:
                subseg = torch.softmax(subseg, dim=1)
                # _, subseg = torch.max(subseg, dim=1)

            segmentation[:, i:i + train_size[0], :, :] = subseg.detach().cpu().numpy()

    # corner case the last one:
    subvolume = volume[:, d-train_size[0]:d, h-train_size[1]:h, w-train_size[2]:w]
    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
    subseg, _ = model(subvolume.unsqueeze(0))

    if class_no == 2:
        subseg = torch.sigmoid(subseg)
        # subseg = (subseg > 0.9).float()
    else:
        subseg = torch.softmax(subseg, dim=1)
        # _, subseg = torch.max(subseg, dim=1)
    segmentation[:, d-train_size[0]:d, h-train_size[1]:h, w-train_size[2]:w] = subseg.squeeze(0).detach().cpu().numpy()
    return segmentation


def segmentation_one_case_one_model(model_path,
                                    data_path,
                                    save_path,
                                    sizes,
                                    classno=2):

    model = torch.load(model_path)
    test_data_path = data_path + '/imgs'
    test_label_path = data_path + '/lbls'

    all_cases = [os.path.join(test_data_path, f) for f in listdir(test_data_path)]
    all_cases.sort()
    all_labels = [os.path.join(test_label_path, f) for f in listdir(test_label_path)]
    all_labels.sort()

    for each_case, each_label in zip(all_cases, all_labels):

        volume_nii = nib.load(each_case)
        volume = volume_nii.get_fdata()

        volume = np.transpose(volume, (2, 0, 1))
        volume = np.expand_dims(volume, axis=0)

        save_name_ext = os.path.split(each_case)[-1]
        save_name = os.path.splitext(save_name_ext)[0]
        save_name_nii = save_name + '_seg.nii.gz'

        segmentation_np = None

        # ensemble testing on different sizes
        for each_size in sizes:
            segmentation_np_current = segment_whole_volume(model, volume, each_size, classno)
            segmentation_np_current = segmentation_np_current[0, :, :, :]
            segmentation_np_current = np.transpose(segmentation_np_current, (1, 2, 0))
            segmentation_np += segmentation_np_current

        segmentation_np = segmentation_np / len(sizes)

        if classno == 2:
            segmentation_np = np.where(segmentation_np > 0.5, 1, 0)
        else:
            segmentation_np = np.argmax(segmentation_np, axis=1)

        segmentation_nii = nib.Nifti1Image(segmentation_np,
                                           volume_nii.affine,
                                           volume_nii.header)

        save_path_nii = os.path.join(save_path, save_name_nii)
        nib.save(segmentation_nii, save_path_nii)

        print(save_path_nii + ' is saved.\n')


if __name__ == '__main__':
    model_path = '/home/moucheng/projects_codes/Results/airway/Turkish/20220202/sup_unet_e1_l0.0001_b2_w16_s4000_d3_z32_x256/trained_models/sup_unet_e1_l0.0001_b2_w16_s4000_d3_z32_x256_3998.pt'
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Mixed/test_nii'
    save_path = '/home/moucheng/projects_codes/Results/airway/Turkish/20220202/sup_unet_e1_l0.0001_b2_w16_s4000_d3_z32_x256/segmentation'

    segmentation_one_case_one_model(model_path,
                                    data_path,
                                    save_path,
                                    sizes=[[12, 256, 256],
                                           [128, 256, 256]],
                                    classno=2)











