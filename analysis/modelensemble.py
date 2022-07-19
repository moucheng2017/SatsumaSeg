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


def average_model_weights(model_path,
                           step_range):

    for each_model in model_path:
        model = torch.load(each_model)
        mds = model.state_dict()
