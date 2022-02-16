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


def ensemble(path, savepath):
    all_cases = [os.path.join(path, f) for f in listdir(path)]
    all_cases.sort()

    seg_nii = nib.load(all_cases[0])
    seg = seg_nii.get_fdata()
    seg = np.zeros_like(seg)

    for each_case in all_cases:
        seg_nii = nib.load(each_case)
        seg_ = seg_nii.get_fdata()
        seg += seg_

    seg = seg / len(all_cases)
    seg = np.where(seg > 0.5, 1, 0)

    seg = nib.Nifti1Image(seg, seg_nii.affine, seg_nii.header)

    save_name_nii = 'ensemble_lung_masked.nii.gz'
    save_path_nii = os.path.join(savepath, save_name_nii)
    nib.save(seg, save_path_nii)


if __name__ == '__main__':
    ensemble(path='/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/seg',
             savepath='/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/ensemble')