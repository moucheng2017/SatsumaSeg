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


# def adjustcontrast(data,
#                    lung_mask,
#                    bin_number,
#                    adjust_times=0):
#     # outputs = deque()
#     outputs = []
#     if adjust_times == 0:
#         data = normalisation(lung=lung_mask, image=data)
#         return outputs.append(data)
#     else:
#         contrast_augmentation = RandomContrast(bin_range=[10, 250])
#         for i in range(adjust_times):
#             data_ = contrast_augmentation.randomintensity(data)
#             data_ = normalisation(lung=lung_mask, image=data_)
#             outputs.append(data_)
#         return outputs


def adjustcontrast(input,
                   bin_no=10):
    c, d, h, w = np.shape(input)
    image_histogram, bins = np.histogram(input.flatten(), bin_no, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    output = np.interp(input.flatten(), bins[:-1], cdf)
    output = np.reshape(output, (c, d, h, w))
    return output


def seg_d_direction(volume,
                    model,
                    new_size,
                    sliding_window,
                    temperature=2.0,
                    padding_size=100):

    c, d, h, w = volume.size()
    print(volume.size())

    print('volume has ' + str(d) + ' slices')

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    # padding:
    if padding_size > 1:
        seg_padded = np.pad(seg, pad_width=((0, padding_size),
                                            (0, padding_size),
                                            (0, padding_size)), mode='symmetric')

        padded_h, padded_w, padded_d = np.shape(seg_padded)

    for dd in range(0, d-new_size[0]+padding_size, sliding_window):
        img = volume[:, dd:dd+new_size[0], :, :]
        # use sliding windows:
        for h_ in range(0, h-new_size[1], sliding_window):
            for w_ in range(0, w-new_size[2], sliding_window):
                cropped = img[:, :, h_:h_ + new_size[1], w_:w_ + new_size[2]]
                seg_patch, _ = model(cropped)
                seg_patch = torch.sigmoid(seg_patch / temperature)
                seg_patch = seg_patch.squeeze().detach().cpu().numpy()
                if padding_size > 1:
                    seg_padded[h_:h_ + new_size[1], w_:w_ + new_size[2], dd:dd+new_size[0]] = np.transpose(seg_patch, (1, 2, 0))
                else:
                    seg[h_:h_ + new_size[1], w_:w_ + new_size[2], dd:dd+new_size[0]] = np.transpose(seg_patch, (1, 2, 0))
        print('d plane slice ' + str(dd) + ' is done...')

    if padding_size > 1:
        seg = seg_padded[padding_size:padded_h-padding_size, padding_size:padded_w-padding_size, padding_size:padded_d-padding_size]

    return seg


def seg_h_direction(volume,
                    model,
                    new_size,
                    sliding_window,
                    temperature=2.0,
                    padding_size=100
                    ):
    '''
    new_size: d x h x w
    '''
    # new_size[1]
    c, d, h, w = volume.size()
    print('volume has ' + str(h) + ' slices')

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    # padding:
    if padding_size > 1:
        seg_padded = np.pad(seg, pad_width=((0, padding_size),
                                            (0, padding_size),
                                            (0, padding_size)), mode='symmetric')
        padded_h, padded_w, padded_d = np.shape(seg_padded)

    for hh in range(0, h-new_size[1]+padding_size, sliding_window):
        img = volume[:, :, hh:hh+new_size[1], :]
        # use sliding windows:
        for d_ in range(0, d-new_size[0], sliding_window):
            for w_ in range(0, w-new_size[2], sliding_window):
                cropped = img[:, d_:d_ + new_size[0], :, w_:w_ + new_size[2]] # 1 x 320 x 5 x 320, C x D x H x W
                cropped = cropped.permute(0, 2, 1, 3) # 1 x 5 x 320 x 320, C x H x D x W
                seg_patch, _ = model(cropped)
                seg_patch = torch.sigmoid(seg_patch / temperature)
                seg_patch = seg_patch.squeeze().detach().cpu().numpy() # H x D x W
                # seg[hh:hh + new_size[1], w_:w_ + new_size[2], d_:d_ + new_size[0]] = seg_patch
                if padding_size > 1:
                    seg_padded[hh:hh + new_size[1], w_:w_ + new_size[2], d_:d_+new_size[0]] = np.transpose(seg_patch, (0, 2, 1))
                else:
                    seg[hh:hh + new_size[1], w_:w_ + new_size[2], d_:d_+new_size[0]] = np.transpose(seg_patch, (0, 2, 1))
        print('h plane slice ' + str(hh) + ' is done...')
    if padding_size > 1:
        seg = seg_padded[padding_size:padded_h - padding_size, padding_size:padded_w - padding_size, padding_size:padded_d - padding_size]
    return seg


def seg_w_direction(volume,
                    model,
                    new_size,
                    sliding_window,
                    temperature=2.0,
                    padding_size=100
                    ):
    '''
    new_size: d x h x w
    '''
    c, d, h, w = volume.size()
    print('volume has ' + str(w) + ' slices')

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    # padding:
    if padding_size > 1:
        seg_padded = np.pad(seg, pad_width=((0, padding_size),
                                            (0, padding_size),
                                            (0, padding_size)), mode='symmetric')

        padded_h, padded_w, padded_d = np.shape(seg_padded)

    for ww in range(0, w-new_size[2]+padding_size, sliding_window):
        img = volume[:, :, :, ww:ww+new_size[2]]
        # use sliding windows:
        for d_ in range(0, d-new_size[0], sliding_window):
            for h_ in range(0, h-new_size[1], sliding_window):
                cropped = img[:, d_:d_ + new_size[0], h_:h_ + new_size[1], :] # 1 x 320 x 320 x 5, C x D x H x W
                cropped = cropped.permute(0, 3, 1, 2) # 1 x 5 x 320 x 320, C x W x D x H
                # print(np.shape(cropped))
                seg_patch, _ = model(cropped)
                seg_patch = torch.sigmoid(seg_patch / temperature)
                seg_patch = seg_patch.squeeze().detach().cpu().numpy() # W x D x H
                # print(np.shape(seg_patch))
                if padding_size > 1:
                    seg_padded[h_:h_ + new_size[1], ww:ww + new_size[2], d_:d_+new_size[0]] = np.transpose(seg_patch, (2, 0, 1))
                else:
                    seg[h_:h_ + new_size[1], ww:ww + new_size[2], d_:d_+new_size[0]] = np.transpose(seg_patch, (2, 0, 1))

        print('w plane slice ' + str(ww) + ' is done...')
    if padding_size > 1:
        seg = seg_padded[padding_size:padded_h - padding_size, padding_size:padded_w - padding_size, padding_size:padded_d - padding_size]
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
              sliding_window=2,
              temperature=2.0,
              padding=200,
              bin_number=100):

    d = [new_size_d, new_size_h, new_size_h]
    h = [new_size_w, new_size_d, new_size_h]
    w = [new_size_w, new_size_h, new_size_d]

    # nii --> np:
    data = nii2np(test_data_path)
    lung = nii2np(lung_path)

    # load model:
    model = torch.load(model_path)
    model.eval()

    # normalisation:
    data = normalisation(lung, data)

    # contrast augmentation:
    data = adjustcontrast(data, bin_number)
    data = normalisation(lung, data)

    # np --> tensor
    data = np2tensor(data)

    # segmentation on each plane direction:
    output_w = seg_w_direction(data, model, w, sliding_window, temperature, padding)
    output_h = seg_h_direction(data, model, h, sliding_window, temperature, padding)
    output_d = seg_d_direction(data, model, d, sliding_window, temperature, padding)

    # ensemble 2.5 D approach
    output_prob = (output_d + output_h + output_w) / 3

    # apply lung mask
    lung = np.transpose(lung.squeeze(), (1, 2, 0))
    output_prob = output_prob*lung

    # threshold the network
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

    model_path = '/home/moucheng/projects_codes/Results/airway/2022_06_04/OrthogonalSup2D_e1_l0.001_b5_w24_s5000_r0.001_z5_h448_w448_t2.0/trained_models/'
    model_name = 'OrthogonalSup2D_e1_l0.001_b5_w24_s5000_r0.001_z5_h448_w448_t2.0_2000.pt'
    model_path_full = model_path + model_name

    new_size_d = 5
    new_resolution_w = 448
    new_resolution_h = 448
    threshold = 0.4
    sliding_window = 5
    temperature = 2.0
    padding = 100
    bin_number = 100

    imgs_folder = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/nifti'
    imgs = os.listdir(imgs_folder)
    imgs = [os.path.join(imgs_folder, img) for img in imgs]
    imgs.sort()

    lungmask_folder = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/lungmask'
    lungs = os.listdir(lungmask_folder)
    lungs = [os.path.join(lungmask_folder, lung) for lung in lungs]
    lungs.sort()

    seg_folder = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/seg'
    os.makedirs(seg_folder, exist_ok=True)

    for img_path, lung_path in zip(imgs, lungs):
        all_imgs = os.listdir(img_path)
        all_lungs = os.listdir(lung_path)
        all_imgs = [os.path.join(img_path, file) for file in all_imgs]
        all_lungs = [os.path.join(lung_path, file) for file in all_lungs]
        all_imgs.sort()
        all_lungs.sort()
        save_path = img_path.split('/')[-1]
        save_path = os.path.join(seg_folder, save_path)
        for img, lung in zip(all_imgs, all_lungs):
            # seg_path_name = store_case_path + '/' + each_file[:-7] + '_lunglabel.nii.gz'
            filename = img.split('/')[-1]
            save_name = filename + '_seg.nii.gz'
            save_name_prob = filename + '_prob.nii.gz'
            segmentation, probability = segment2D(img,
                                                  lung,
                                                  model_path_full,
                                                  threshold,
                                                  new_size_d,
                                                  new_resolution_h,
                                                  new_resolution_w,
                                                  sliding_window,
                                                  temperature,
                                                  padding,
                                                  bin_number)
            save_seg(save_path, save_name, img, segmentation)
            save_seg(save_path, save_name_prob, img, probability)

    # save_path = '/home/moucheng/PhD/2022_12_Clinical/orthogonal2d/preliminary/seg_new'
    # data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/imgs/' + case + '.nii.gz'
    # lung_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/lung/' + case + '_lunglabel.nii.gz'
    # model_path = '/home/moucheng/projects_codes/Results/cluster/Results/airway/2022_06_07/OrthogonalSup2DFast_e1_l0.001_b5_w48_s5000_r0.01_z5_h448_w448_t2.0/trained_models/'
    # model_name = 'OrthogonalSup2DFast_e1_l0.001_b5_w48_s5000_r0.01_z5_h448_w448_t2.0_2800.pt'
    # model_path_full = model_path + model_name
    # save_name = case + '_aug_seg2Dorthogonal_t' + str(threshold) + '_h' + str(new_resolution_h) + '_w' + str(new_resolution_w) + '_s' + str(sliding_window) + '_temp' + str(temperature) + '_p' + str(padding) + '_b' + str(bin_number) + '.nii.gz'
    # save_name_prob = case + '_aug_prob2Dorthogonal_t' + str(threshold) + '_h' + str(new_resolution_h) + '_w' + str(new_resolution_w) + '_s' + str(sliding_window) + '_temp' + str(temperature) + '_p' + str(padding) + '_b' + str(bin_number) + '.nii.gz'
    #
    # segmentation, probability = segment2D(data_path,
    #                                       lung_path,
    #                                       model_path_full,
    #                                       threshold,
    #                                       new_size_d,
    #                                       new_resolution_h,
    #                                       new_resolution_w,
    #                                       sliding_window,
    #                                       temperature,
    #                                       padding,
    #                                       bin_number)
    #
    # save_seg(save_path, save_name, data_path, segmentation)
    # save_seg(save_path, save_name_prob, data_path, probability)
    #
    # print(np.shape(segmentation))
    # print('End')
