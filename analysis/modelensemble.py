import nibabel
import torch
import random
import numpy as np
import os
import argparse
import glob
import tifffile as tiff

import errno
import imageio

from PIL import Image

import numpy.ma as ma

from dataloaders.Dataloader import RandomContrast
from collections import deque

import matplotlib.pyplot as plt

from Models2DOrthogonal import Unet2DMultiChannel


def average_model_weights(model_width,
                          model_path,
                          step_lower,
                          step_upper):

    all_models = os.listdir(model_path)
    all_models.sort()
    all_models = [os.path.join(model_path, model_name) for model_name in all_models]

    avg_weights = all_models[0]
    model = Unet2DMultiChannel(in_ch=1, width=24, output_channels=1)
    model.to('cuda')
    checkpoint = torch.load(avg_weights)
    avg_weights = model.load_state_dict(checkpoint['model_state_dict'])

    for each_model in all_models:

        model = Unet2DMultiChannel(in_ch=1,
                                   width=model_width,
                                   output_channels=1)
        model.to('cuda')
        checkpoint = torch.load(each_model)
        # weights =
        # model.load_state_dict(checkpoint['model_state_dict'])

