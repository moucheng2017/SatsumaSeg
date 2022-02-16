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
from dataloaders.Dataloader import RandomContrast
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
    # print(np.shape(volume))
    c, d, h, w = np.shape(volume)
    volume[volume < -1000.0] = -1000.0
    volume[volume > 500.0] = 500.0
    volume = (volume - np.nanmean(volume)) / np.nanstd(volume)
    segmentation = np.zeros_like(volume)
    model.eval()

    ratio_h = 10
    # Loop through the whole volume:

    if full_resolution is False:
        for i in range(0, d - train_size[0], train_size[0]//train_size[0]):
            for j in range(0, h - train_size[1], train_size[1]//train_size[1]):
                for k in range(0, w - train_size[2], train_size[2]//train_size[2]):
                    subvolume = volume[:, i:i+train_size[0], j:j+train_size[1], k:k+train_size[2]]
                    # subvolume = RandomContrast(bin_range=[10, 50]).randomintensity(subvolume)
                    subvolume = (subvolume - np.nanmean(subvolume)) / np.nanstd(subvolume)
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
            subvolume = (subvolume - np.nanmean(subvolume)) / np.nanstd(subvolume)
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
    subvolume = (subvolume - np.nanmean(subvolume)) / np.nanstd(subvolume)
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
                                    size,
                                    classno=2):

    model = torch.load(model_path)
    test_data_path = data_path + '/imgs'
    test_label_path = data_path + '/lbls'

    all_cases = [os.path.join(test_data_path, f) for f in listdir(test_data_path)]
    all_cases.sort()
    all_labels = [os.path.join(test_label_path, f) for f in listdir(test_label_path)]
    all_labels.sort()

    for each_case, each_label in zip(all_cases, all_labels):

        # print(each_case)
        volume_nii = nib.load(each_case)
        volume = volume_nii.get_fdata()

        saved_segmentation = np.zeros_like(volume)

        volume = np.transpose(volume, (2, 0, 1))
        volume = np.expand_dims(volume, axis=0)

        save_name_ext = os.path.split(each_case)[-1]
        save_name = os.path.splitext(save_name_ext)[0]
        # save_name_nii = save_name + '_seg.nii.gz'
        save_name_nii = save_name + '_test_d' + str(size[0]) + '_r' + str(size[1]) + '_ratio10.seg.nii.gz'
        # segmentation_np = None

        # ensemble testing on different sizes
        # for each_size in sizes:
        segmentation_np = segment_whole_volume(model, volume, size, classno)
        segmentation_np = segmentation_np[0, :, :, :]
        # segmentation_np = np.transpose(segmentation_np, (1, 2, 0))
        # segmentation_np += segmentation_np_current

        # segmentation_np = segmentation_np / len(sizes)

        if classno == 2:
            segmentation_np = np.where(segmentation_np > 0.5, 1, 0)
        else:
            segmentation_np = np.argmax(segmentation_np, axis=1)

        h, w, d = np.shape(saved_segmentation)

        # print(np.shape(segmentation_np))
        # print(np.shape(saved_segmentation))

        for dd in range(d):
            saved_segmentation[:, :, dd] = segmentation_np[dd, :, :]

        segmentation_nii = nib.Nifti1Image(saved_segmentation,
                                           volume_nii.affine,
                                           volume_nii.header)

        save_path_nii = os.path.join(save_path, save_name_nii)
        nib.save(segmentation_nii, save_path_nii)

        print(save_path_nii + ' is saved.\n')


if __name__ == '__main__':
    model_path = '/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/trained_models/' \
                 'sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384_3999.pt'

    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Mixed/test'

    save_path = '/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/model3999'

    segmentation_one_case_one_model(model_path,
                                    data_path,
                                    save_path,
                                    size=[16, 480, 480],
                                    classno=2)












