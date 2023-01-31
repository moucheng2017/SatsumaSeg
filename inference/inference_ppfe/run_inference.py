import os
import csv
import numpy as np
import pathlib
import scipy
import sys
import argparse
import nibabel as nib
import torch
from pathlib import Path
sys.path.append('../..')
import matplotlib.pyplot as plt
from skimage.transform import resize

# This prints out a grid: rows are different images, each row has three columns of image, label, and overlapped image

parser = argparse.ArgumentParser('Run inference on PPFE')
parser.add_argument('--img_source', type=str, help='source file', default='/home/moucheng/projects_data/PPFE_HipCT/processed/img_volume.npy')
parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/PhD/Clinical/ppfe/3d_ppfe_sup_binary_res128/Unet3D_l_0.0003_b2_w8_d4_i14500_l2_0.0005_c_True_t1.0/trained_models/Unet3D_l_0.0003_b2_w8_d4_i14500_l2_0.0005_c_True_t1.0_ema.pt')
parser.add_argument('--save_path', type=str, help='save path', default='/home/moucheng/projects_data/PPFE_HipCT/processed/inference')
parser.add_argument('--new_dim', type=int, help='new dimension', default=128)
args = parser.parse_args()


if __name__ == '__main__':

    # Read the file
    img_volume = np.load(args.img_source)
    img = np.asfarray(img_volume)
    d, h, w = np.shape(img)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    model = torch.load(args.model_source).cuda()

    new_dim = args.new_dim

    d_start = (d - (d // new_dim)*new_dim) // 2
    d_end = d_start + (d // new_dim)*new_dim

    h_start = (h - (h // new_dim)*new_dim) // 2
    h_end = h_start + (h // new_dim)*new_dim

    w_start = (w - (w // new_dim)*new_dim) // 2
    w_end = w_start + (w // new_dim)*new_dim

    img = img[d_start:d_end, h_start:h_end, w_start:w_end]
    seg = np.zeros_like(img)

    img = (img - np.nanmean(img) + 1e-10) / (np.nanstd(img) + 1e-10)

    # imgs_d = np.split(img, d // new_dim, axis=0)

    count = 0

    # for i, each_img_d in enumerate(imgs_d):
    #     imgs_d_h = np.split(each_img_d, h // new_dim, axis=1)
    #     for j, each_img_h in enumerate(imgs_d_h):
    #         imgs_d_h_w = np.split(each_img_h, w // new_dim, axis=2)
    #         for k, each_img_w in enumerate(imgs_d_h_w):
    #             d_, h_, w_ = np.shape(each_img_w)
    #
    #             assert d_ == new_dim
    #             assert h_ == new_dim
    #             assert w_ == new_dim
    #
    #             seg_ = torch.from_numpy(each_img_w).cuda().unsqueeze(0).unsqueeze(0).float()
    #             seg_ = model(seg_)
    #             seg_ = seg_.get('segmentation')
    #             _, seg_ = torch.max(seg_, dim=1)
    #             seg_ = seg_.detach().cpu().numpy()
    #             seg[i*new_dim:(i+1)*new_dim, j*new_dim:(j+1)*new_dim, k*new_dim:(k+1)*new_dim] = seg_
    #
    #             count += 1
    #             print(count)
    #
    #             # seg_name = str(args.new_dim) + '_' + str(count) + '_segmentation.npy'
    #             # save_file = os.path.join(args.save_path, seg_name)
    #             # np.save(seg_, save_file)
    #             # del each_img_w
    #             # del seg_

    for i in range(0, d - new_dim, 16):
        for j in range(0, h - new_dim, 16):
            for k in range(0, w - new_dim, 16):
                input_volume = img[i:i+new_dim, j:j+new_dim, k:k+new_dim]
                seg_ = torch.from_numpy(input_volume).cuda().unsqueeze(0).unsqueeze(0).float()
                seg_ = model(seg_)
                seg_ = seg_.get('segmentation')
                _, seg_ = torch.max(seg_, dim=1)
                seg_ = seg_.detach().cpu().numpy()
                seg[i:i+new_dim, j:j+new_dim, k:k+new_dim] = seg_
                count += 1
                print(count)

    seg = nib.Nifti1Image(seg, affine=np.eye(4))
    seg_name = str(args.new_dim) + '_segmentation.nii'
    save_file = os.path.join(args.save_path, seg_name)
    nib.save(seg, save_file)
    print('Done')




