import os

import matplotlib.pyplot as plt
import numpy as np

from libs.Augmentations import *
from PIL import Image

import pathlib as Path


if __name__ == '__main__':
    img_path = '/home/moucheng/projects_data/PPFE_HipCT/processed/imgs'
    all_slices = os.listdir(img_path)
    all_slices.sort()
    all_slices = [os.path.join(img_path, i) for i in all_slices]


