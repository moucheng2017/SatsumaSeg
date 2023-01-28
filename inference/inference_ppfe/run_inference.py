import os
import csv
import numpy as np
import pathlib
import scipy
import sys
import argparse

import torch

sys.path.append('../..')
import matplotlib.pyplot as plt
from skimage.transform import resize

# This prints out a grid: rows are different images, each row has three columns of image, label, and overlapped image

parser = argparse.ArgumentParser('Run inference on PPFE')
parser.add_argument('--img_source', type=str, help='source file', default='/home/moucheng/projects_data/PPFE_HipCT/processed/res256_type3/labelled/imgs')
parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/model')
parser.add_argument('--save_path', type=str, help='save path', default='/home/moucheng/projects_data/PPFE_HipCT/processed/res256_type3/labelled/lbls')
parser.add_argument('--new_dim', type=int, help='new dimension', default=224)
args = parser.parse_args()


if __name__ == '__main__':

    # Read the file
    img_volume = np.load(args.img_source)
    img = np.asfarray(img_volume)
    d, h, w = np.shape(img)

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

    assert d % new_dim == 0
    assert h % new_dim == 0
    assert w % new_dim == 0

    imgs_d = np.split(img, d // new_dim, axis=0)
    count = 0

    for each_img_d in imgs_d:
        imgs_d_h = np.split(each_img_d, h // new_dim, axis=1)
        for each_img_h in imgs_d_h:
            imgs_d_h_w = np.split(each_img_h, w // new_dim, axis=2)
            for each_img_w in imgs_d_h_w:
                d_, h_, w_ = np.shape(each_img_w)

                print('The shape of the saved subvolume. Depth: {:.1f}; ' 'Height: {:.1f}; ' 'Width: {:.1f}'.format(d_, h_, w_))
                assert d_ == new_dim
                assert h_ == new_dim
                assert w_ == new_dim

                # run inference
                seg_ = torch.from_numpy(each_img_w).cuda().unsqueeze(1)
                seg_ = model(seg_)
                _, seg_ = torch.max(seg_, dim=1)



