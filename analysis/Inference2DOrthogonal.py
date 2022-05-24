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

import matplotlib.pyplot as plt

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
    data = np.array(data, dtype='float32')
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


def seg_d_direction(volume,
                    model,
                    new_size):

    c, d, h, w = volume.size()
    print('volume has ' + str(d) + ' slices')

    # location_list = {
    #     "left_up": [0, 0],
    #     "center_up": [0, (w-new_size[2]) // 2],
    #     "right_up": [0, w-new_size[2]],
    #     "left_middle": [(h - new_size[1])//2, 0],
    #     "center_middle": [(h - new_size[1])//2, (w-new_size[2]) // 2],
    #     "right_middle": [(h - new_size[1])//2, w-new_size[2]],
    #     "left_bottom": [h-new_size[1], 0],
    #     "center_bottom": [h-new_size[1], (w-new_size[2]) // 2],
    #     "right_bottom": [h-new_size[1], w-new_size[2]]
    # }

    # location_list = {
    #     "left_up": [0, 0],
    #     "right_up": [0, w-new_size],
    #     "center_middle": [(h - new_size)//2, (w-new_size) // 2],
    #     "left_bottom": [h-new_size, 0],
    #     "right_bottom": [h-new_size, w-new_size]
    # }

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    # print(seg.size())

    for dd in range(0, d-new_size[0]):
        img = volume[:, dd:dd+new_size[0], :, :] # c x dd x h x w
        # # use 9 samples:
        # for each_location, each_coordiate in location_list.items():
        #     cropped = img[:, :, each_coordiate[0]:each_coordiate[0] + new_size, each_coordiate[1]:each_coordiate[1] + new_size]
        #     cropped = torch.unsqueeze(cropped, dim=0)
        #     seg_patch, _ = model(cropped)
        #     seg_patch = torch.sigmoid(seg_patch)
        #     seg[each_coordiate[0]:each_coordiate[0] + new_size, each_coordiate[1]:each_coordiate[1] + new_size, dd] = seg_patch.detach().cpu().numpy()

        # use sliding windows:
        for h_ in range(0, h-new_size[1], new_size[1] // 2):
            for w_ in range(0, w-new_size[2], new_size[2] // 2):
                cropped = img[:, h_:h_ + new_size[1], w_:w_ + new_size[2]]
                cropped = torch.unsqueeze(cropped, dim=0)
                seg_patch, _ = model(cropped)
                seg_patch = torch.sigmoid(seg_patch)
                seg_patch = seg_patch.squeeze().detach().cpu().numpy()
                seg[h_:h_ + new_size[1], w_:w_ + new_size[2], dd:dd+new_size[0]] = np.transpose(seg_patch, (1, 2, 0))
        print('slice ' + str(dd) + ' is done...')

    return seg


def seg_h_direction(volume,
                    model,
                    new_size
                    ):
    '''
    new_size: d x h x w
    '''
    c, d, h, w = volume.size()
    print('volume has ' + str(d) + ' slices')

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    for hh in range(0, h-new_size[1]):
        img = volume[:, :, hh:hh+new_size[1], :]
        # use sliding windows:
        for d_ in range(0, d-new_size[0], new_size[0] // 2):
            for w_ in range(0, w-new_size[2], new_size[2] // 2):
                cropped = img[:, d_:d_ + new_size[0], :, w_:w_ + new_size[2]]
                cropped = np.transpose(cropped, axes=(1, 0, 2))
                cropped = torch.unsqueeze(cropped, dim=0)
                seg_patch, _ = model(cropped)
                seg_patch = torch.sigmoid(seg_patch)
                seg_patch = seg_patch.squeeze().detach().cpu().numpy()
                seg[hh:hh + new_size[1], w_:w_ + new_size[2], d_:d_+new_size[0]] = np.transpose(seg_patch, (1, 2, 0))
        print('slice ' + str(hh) + ' is done...')

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
              new_size_w):

    d = [new_size_d, new_size_h, new_size_w]
    h = [new_size_h, new_size_d, new_size_w]

    # nii --> np:
    data = nii2np(test_data_path)
    lung = nii2np(lung_path)

    # load model:
    model = torch.load(model_path)
    model.eval()

    data = np2tensor(data)
    output_d = seg_d_direction(data, model, d)
    output_h = seg_h_direction(data, model, h)
    output = (output_d + output_h) / 2

    lung = np.transpose(lung.squeeze(), (1, 2, 0))
    output = output*lung
    output = np.where(output > threshold, 1, 0)

    return np.squeeze(output)


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
    new_size_h = 320
    new_size_w = 320
    threshold = 0.4

    save_path = '/home/moucheng/PhD/2022_12_Clinical/orthogonal2d/preliminary/seg'
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/imgs/' + case + '.nii.gz'
    lung_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/lung/' + case + '_lunglabel.nii.gz'
    model_path = '/home/moucheng/PhD/2022_12_Clinical/orthogonal2d/preliminary/airway/2022_05_13/OrthogonalSup2D_e1_l0.001_b6_w64_s5000_r0.01_z5_t2.0/trained_models/'
    model_name = 'OrthogonalSup2D_e1_l0.001_b6_w64_s5000_r0.01_z5_t2.0_2000.pt'
    model_path_full = model_path + model_name
    save_name = case + '_seg2D_t' + str(threshold) + '.nii.gz'

    segmentation = segment2D(data_path,
                             lung_path,
                             model_path_full,
                             threshold,
                             new_size_d,
                             new_size_h,
                             new_size_w)

    save_seg(save_path, save_name, data_path, segmentation)
    print(np.shape(segmentation))
    print('End')



