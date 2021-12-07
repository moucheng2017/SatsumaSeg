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


# def crop_slice(ct_scan, label_scan, left, right, top, bottom, input_dim, sample_position, ):
#     label = label_scan[sample_position:sample_position+input_dim, top:bottom, left:right]
#     label = np.transpose(label, (2, 0, 1))
#
#     scan = ct_scan[sample_position:sample_position+input_dim, top:bottom, left:right]
#     scan = np.transpose(scan, (2, 0, 1))
#
#     return scan, label


# def segment_all_patches(image, label, new_size, new_dim, height, width, slice_location, labelled_flag):
#
#     left = (width - new_size) // 2
#     right = left + new_size
#     top = (height - new_size) // 2
#     bottom = (height - new_size) // 2 + new_size
#     img, lbl = crop_slice(image, label, left, right, top, bottom, new_dim, slice_location)
#     segment_patch(img, lbl, slice_location, 6, labelled_flag)


# def segment_patch(image, label, slice_location, patch_lodation, labelled_flag):
#     if labelled_flag is True:
#         foreground_pixels = np.sum(label)
#         if foreground_pixels > 10:
#             img_slice_store_name = case_index + '_slice_' + str(slice_location) + '_' + str(patch_lodation) + '.npy'
#             gt_slice_store_name = case_index + '_gt_' + str(slice_location) + '_' + str(patch_lodation) + '.npy'
#             img_slice_store_name = save_img_path + '/' + img_slice_store_name
#             gt_slice_store_name = save_lbl_path + '/' + gt_slice_store_name
#             np.save(img_slice_store_name, image)
#             np.save(gt_slice_store_name, label)
#         else:
#             pass
#     else:
#         img_slice_store_name = case_index + '_slice_' + str(slice_location) + '_' + str(patch_lodation) + '.npy'
#         gt_slice_store_name = case_index + '_gt_' + str(slice_location) + '_' + str(patch_lodation) + '.npy'
#         img_slice_store_name = save_img_path + '/' + img_slice_store_name
#         gt_slice_store_name = save_lbl_path + '/' + gt_slice_store_name
#         np.save(img_slice_store_name, image)
#         np.save(gt_slice_store_name, label)


def prepare_data(data_path, lbl_path, save_img_path, save_lbl_path, case_index, labelled=True):
    # make directory first:
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    # clean up the directory:
    files = os.listdir(save_img_path)
    if len(files) > 0:
        for f in files:
            os.remove(os.path.join(save_img_path, f))

    files = os.listdir(save_lbl_path)
    if len(files) > 0:
        for f in files:
            os.remove(os.path.join(save_lbl_path, f))

    #  normalising intensities:
    data = nib.load(data_path)
    data = data.get_fdata()
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    data = (data - np.nanmean(data)) / np.nanstd(data)
    lbl = nib.load(lbl_path)
    lbl = lbl.get_fdata()

    if labelled is False:
        lbl = 100*np.ones_like(lbl)
    else:
        pass

    new_data = data[:, :, :]
    new_data = np.transpose(new_data, (2, 0, 1))
    new_data = np.expand_dims(new_data, axis=0)

    new_lbl = lbl[:, :, :]
    new_lbl = np.transpose(new_lbl, (2, 0, 1))
    new_lbl = np.expand_dims(new_lbl, axis=0)

    img_slice_store_name = case_index + '_volume.npy'
    gt_slice_store_name = case_index + '_label.npy'
    img_slice_store_name = save_img_path + '/' + img_slice_store_name
    gt_slice_store_name = save_lbl_path + '/' + gt_slice_store_name
    np.save(img_slice_store_name, new_data)
    np.save(gt_slice_store_name, new_lbl)


if __name__ == '__main__':

    # labelled train:
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllRaw/1841A.nii.gz'
    lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllSeg/1841A_seg.nii.gz'
    save_img_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/labelled/patches/'
    save_lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/labelled/labels/'
    case_index = '1841A'
    prepare_data(data_path, lbl_path, save_img_path, save_lbl_path, case_index, labelled=True)

    # unlabelled train:
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllRaw/6357A.nii.gz'
    lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllSeg/6357A_seg.nii.gz'
    save_img_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/unlabelled/patches/'
    save_lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/unlabelled/labels/'
    case_index = '6357A'
    prepare_data(data_path, lbl_path, save_img_path, save_lbl_path, case_index, labelled=False)

    # # unlabelled train:
    # data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllRaw/9731A.nii.gz'
    # lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllSeg/9731A_seg.nii.gz'
    # save_img_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/unlabelled/patches'
    # save_lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/unlabelled/labels'
    # case_index = '9731A'
    # prepare_data(data_path, lbl_path, new_dim, save_img_path, save_lbl_path, case_index, labelled=False)

    # validate:
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllRaw/6357B.nii.gz'
    lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllSeg/6357B_seg.nii.gz'
    save_img_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/validate/patches/'
    save_lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/validate/labels/'
    case_index = '6357B'
    prepare_data(data_path, lbl_path, save_img_path, save_lbl_path, case_index, labelled=True)

    # test:
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllRaw/6610A.nii.gz'
    lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/AllSeg/6610A_seg.nii.gz'
    save_img_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/test/patches/'
    save_lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/mismatch_exp/test/labels/'
    case_index = '6610A'
    prepare_data(data_path, lbl_path, save_img_path, save_lbl_path, case_index, labelled=True)

print('End')





