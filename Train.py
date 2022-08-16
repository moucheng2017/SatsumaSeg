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
from libs.Utils import train_base

# training options control panel:
from TrainArguments import parser

# training data loader:
from libs.DataloaderOrthogonal import getData

from libs import TrainBase


def main():
    global args
    args = parser.parse_args()

    if args.seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.unlabelled == 0:
        # supervised learning:
        model = Unet(in_ch=args.input_dim,
                     width=args.width,
                     depth=args.depth,
                     classes=args.output_dim,
                     norm='in',
                     side_output=False)

        model_name = 'Unet_l' + str(args.lr) + \
                   '_b' + str(args.batch) + \
                   '_w' + str(args.width) + \
                   '_d' + str(args.depth) + \
                   '_i' + str(args.iterations) + \
                   '_l2_' + str(args.l2) + \
                   '_c_' + str(args.contrast) + \
                   '_n_' + str(args.norm) + \
                   '_t' + str(args.temp) + \
                   '_m_' + str(args.lung_mask)
    else:
        # supervised learning plus pseudo labels:
        model = UnetBPL(in_ch=args.input_dim,
                        width=args.width,
                        depth=args.depth,
                        out_ch=args.output_dim,
                        norm='in',
                        ratio=8,
                        detach=args.detach)

        model_name = 'BPUnet_l' + str(args.lr) + \
                   '_b' + str(args.batch) + \
                   '_u' + str(args.unlabelled) + \
                   '_w' + str(args.width) + \
                   '_d' + str(args.depth) + \
                   '_i' + str(args.iterations) + \
                   '_l2_' + str(args.l2) + \
                   '_c_' + str(args.contrast) + \
                   '_n_' + str(args.norm) + \
                   '_t' + str(args.temp) + \
                   '_m_' + str(args.lung_mask) + \
                   '_de_' + str(args.detach) + \
                   '_mu' + str(args.mu) + \
                   '_sig' + str(args.sigma) + \
                   '_a' + str(args.alpha) + \
                   '_w' + str(args.warmup)

    # data loader:


    # running loop:
    # for step in range(args.iterations):
    #     outputs = train_base(labelled_img=,
    #                          labelled_label,
    #                          labelled_lung,
    #                          model,
    #                          unlabelled_img=None,
    #                          t=2.0,
    #                          prior_mu=0.4,
    #                          prior_logsigma=0.1,
    #                          augmentation_cutout=True,
    #                          apply_lung_mask=True)










