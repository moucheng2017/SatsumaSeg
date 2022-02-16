import numpy as np
import pydicom
import os
import glob
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import nibabel as nib
# import itk as itk
# import dicom2nifti
from skimage import measure, morphology, segmentation


def lungseparation(lung_path, seg_path, save_path):
    lung_nii = nib.load(lung_path)
    lung_mask = lung_nii.get_fdata()

    seg_nii = nib.load(seg_path)
    seg = seg_nii.get_fdata()

    seg_masked = seg*lung_mask

    segmentation_nii = nib.Nifti1Image(seg_masked,
                                       seg_nii.affine,
                                       seg_nii.header)

    save_name_ext = os.path.split(seg_path)[-1]
    save_name = os.path.splitext(save_name_ext)[0]
    save_name_nii = save_name + '_lung_masked.nii.gz'
    save_path_nii = os.path.join(save_path, save_name_nii)
    nib.save(segmentation_nii, save_path_nii)
    print('Done')


if __name__ == "__main__":
    lung_path = '/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/lung_label/Pat25b_lunglabel.nii.gz'
    seg_path = '/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/model3999/Pat25b.nii_test_d16_r448_seg.nii.gz'
    save_path = '/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/seg'
    lungseparation(lung_path,
                   seg_path,
                   save_path)