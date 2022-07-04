import glob
import os
import random

import torch
import numpy as np

import nibabel as nib

from tqdm import tqdm
import numpy.ma as ma


def main(path):
    all_files = os.listdir(path)
    all_files = [os.path.join(path, file) for file in all_files]

    for full_path in tqdm(all_files):
        nifti = nib.load(full_path)
        data = nifti.get_fdata()
        data = np.array(data, dtype='float32')
        d1, d2, d3 = np.shape(data)
        data_padded = np.pad(data, pad_width=((0, 512-d1), (0, 512-d2), (0, 512-d3)), mode='symmetric')
        data_nii = nib.Nifti1Image(data_padded, nifti.affine, nifti.header)
        nib.save(data_nii, full_path)


if __name__ == "__main__":
    all_paths = ['/home/moucheng/projects_data/Pulmonary_data/airway/validate/imgs',
                 '/home/moucheng/projects_data/Pulmonary_data/airway/validate/lbls',
                 '/home/moucheng/projects_data/Pulmonary_data/airway/validate/lung']

    for each_path in all_paths:
        main(each_path)

    print('End')


