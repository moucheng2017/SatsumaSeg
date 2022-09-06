import parser

import torch
# torch.manual_seed(0)
import errno
import numpy as np
# import pandas as pd


import os
from os import listdir
# import Image

import timeit
import torch.nn as nn
from pathlib import Path

# Deterministic training:
import random
import numpy as np
import torch.backends.cudnn as cudnn

# model:
from Models2D import Unet, UnetBPL
from libs.Train import train_base

# data:
from libs.DataloaderOrthogonal import getData

# track the training
from tensorboardX import SummaryWriter


def reproducibility(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def network_intialisation(args):
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
                       '_t' + str(args.temp)

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
                       '_de_' + str(args.detach) + \
                       '_mu' + str(args.mu) + \
                       '_sig' + str(args.sigma) + \
                       '_a' + str(args.alpha) + \
                       '_w' + str(args.warmup)

    return model, model_name


def get_data_simple_wrapper(args):
    data_loaders = getData(data_directory=args.data_path,
                           dataset_name=args.dataset,
                           train_batchsize=args.batch,
                           norm=args.norm,
                           zoom_aug=args.zoom,
                           sampling_weight=args.sampling,
                           contrast_aug=args.contrast,
                           lung_window=args.lung_mask,
                           resolution=512,
                           train_full=True,
                           unlabelled=args.unlabelled)

    return data_loaders


def make_saving_directories(model_name, args):
    save_model_name = model_name
    saved_information_path = '../Results/' + args.dataset + '/' + args.log_tag
    Path(saved_information_path).mkdir(parents=True, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    Path(saved_log_path).mkdir(parents=True, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    Path(saved_model_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)
    return writer, saved_model_path


def get_iterators(data_loaders, args):
    iterator_train_labelled = iter(data_loaders.get('train_loader_l'))
    if args.unlabelled > 0:
        iterator_train_unlabelled = iter(data_loaders.get('train_loader_u'))
        if args.full_train is False:
            iterator_validate = iter(data_loaders.get('val_loader'))
            return {'train labelled': iterator_train_labelled,
                    'train unlabelled': iterator_train_unlabelled,
                    'val': iterator_validate}
        else:
            return {'train labelled': iterator_train_labelled,
                    'train unlabelled': iterator_train_unlabelled}
    else:
        if args.full_train is False:
            iterator_validate = iter(data_loaders.get('val_loader'))
            return {'train labelled': iterator_train_labelled,
                    'val': iterator_validate}
        else:
            return {'train labelled': iterator_train_labelled}


def get_data_dict(dataloader, iterator):
    try:
        data_dict, data_name = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        data_dict, data_name = next(iterator)
    del data_name
    return data_dict


# def get_losses(args, sup_data_dict, unsup_data_dict):
#     if args.unlabelled > 0:
#         # calculate loss for each plane of supervised:
#         # todo: remove lung mask, add loss function calculations
#         # loss_d, train_mean_iu_d_ = train_base(labelled_dict["plane_d"][0], labelled_dict["plane_d"][1], labelled_dict["plane_d"][2], device, model, temp, apply_lung_mask, single_channel_output)








