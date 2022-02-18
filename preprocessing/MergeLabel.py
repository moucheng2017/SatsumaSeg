import glob
import os
# import gzip
# import shutil
# import random
import errno
# # import pydicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# from nipype.interfaces.ants import N4BiasFieldCorrection
from tifffile import imsave


def merge_label_binary(path):
    all_niis = [os.path.join(path, i) for i in os.listdir(path)]
    for nii_path in all_niis:
        nii = nib.load(nii_path)
        niidata = nii.get_fdata()
        niidata[niidata > 0] = 1

        segmentation_nii = nib.Nifti1Image(niidata,
                                           nii.affine,
                                           nii.header)
        nib.save(segmentation_nii, nii_path)
    print('done')


if __name__ == '__main__':
    merge_label_binary('')
