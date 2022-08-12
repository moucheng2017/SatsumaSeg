# This will be the main script for both training with semi supervised learning and supervised learning
# basic libs:
import math
import torch
import timeit
import shutil
from pathlib import Path

# Deterministic training:
import random
import numpy as np
import torch.backends.cudnn as cudnn

# Tracking the training process:
# import wandb
from tensorboardX import SummaryWriter

# model:
from Models2D import Unet, UnetBPL

# training options control panel:
from TrainArguments import parser

# training data loader:
from libs.DataloaderOrthogonal import getData

from libs import TrainBase

# class