import os
import torch
from pathlib import Path

import random
import numpy as np
import torch.backends.cudnn as cudnn

# model:
from Models2D import Unet, UnetBPL

# model
from Models3D import Unet3D

# data:
from libs.Dataloader import getData

from libs.Dataloader3D import getData3D

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
    if args.train.batch_u == 0:
        # supervised learning:
        if args.model.use_3d == 0:
            model = Unet(in_ch=args.model.input_dim,
                         width=args.model.width,
                         depth=args.model.depth,
                         classes=args.model.output_dim,
                         norm='in',
                         side_output=False)

            model_name = 'Unet_l' + str(args.train.lr) + \
                           '_b' + str(args.train.batch) + \
                           '_w' + str(args.model.width) + \
                           '_d' + str(args.model.depth) + \
                           '_i' + str(args.train.iterations) + \
                           '_l2_' + str(args.train.optimizer.weight_decay) + \
                           '_c_' + str(args.train.contrast) + \
                           '_t' + str(args.train.temp)
        else:
            model = Unet3D(in_ch=args.model.input_dim,
                           width=args.model.width,
                           depth=args.model.depth,
                           classes=args.model.output_dim,
                            norm='in',
                            side_output=False)

            model_name = 'Unet3D_l_' + str(args.train.lr) + \
                         '_b' + str(args.train.batch) + \
                         '_w' + str(args.model.width) + \
                         '_d' + str(args.model.depth) + \
                         '_i' + str(args.train.iterations) + \
                         '_l2_' + str(args.train.optimizer.weight_decay) + \
                         '_c_' + str(args.train.contrast) + \
                         '_t' + str(args.train.temp)

    else:

        if args.model.use_3d != 0:
            raise NotImplementedError

        # supervised learning plus pseudo labels:
        model = UnetBPL(in_ch=args.model.input_dim,
                        width=args.model.width,
                        depth=args.model.depth,
                        out_ch=args.model.output_dim,
                        norm='in',
                        ratio=8
                        # detach=args.detach
                        )

        model_name = 'BPUnet_l' + str(args.train.lr) + \
                       '_b' + str(args.train.batch) + \
                       '_u' + str(args.train.batch_u) + \
                       '_w' + str(args.model.width) + \
                       '_d' + str(args.model.depth) + \
                       '_i' + str(args.train.iterations) + \
                       '_l2_' + str(args.train.optimizer.weight_decay) + \
                       '_c_' + str(args.train.contrast) + \
                       '_t' + str(args.train.temp) + \
                       '_mu' + str(args.train.mu) + \
                       '_a' + str(args.train.alpha) + \
                       '_w' + str(args.train.warmup)

    # '_de_' + str(args.detach) + \
    # '_sig' + str(args.sigma) + \
    return model, model_name


def make_saving_directories(model_name, args):
    save_model_name = model_name
    dataset_name = os.path.basename(os.path.normpath(args.dataset.data_dir))
    saved_information_path = '../../Results_' + dataset_name + '/' + args.logger.tag
    Path(saved_information_path).mkdir(parents=True, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    Path(saved_log_path).mkdir(parents=True, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    Path(saved_model_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)
    return writer, saved_model_path


def get_iterators(args):
    if args.model.use_3d == 0:
        data_loaders = getData(data_directory=args.dataset.data_dir,
                               train_batchsize=args.train.batch,
                               zoom_aug=args.train.zoom,
                               gaussian_aug=args.train.gaussian,
                               data_format=args.dataset.data_format,
                               # sampling_weight=args.sampling,
                               contrast_aug=args.train.contrast,
                               unlabelled=args.train.batch_u,
                               output_shape=(args.train.new_size_h, args.train.new_size_w),
                               full_orthogonal=args.train.full_orthogonal)
    else:
        data_loaders = getData3D(data_directory=args.dataset.data_dir,
                               train_batchsize=args.train.batch,
                               zoom_aug=args.train.zoom,
                               gaussian_aug=args.train.gaussian,
                               data_format=args.dataset.data_format,
                               # sampling_weight=args.sampling,
                               contrast_aug=args.train.contrast,
                               unlabelled=args.train.batch_u,
                               output_shape=(args.train.new_size_h, args.train.new_size_w),
                               full_orthogonal=args.train.full_orthogonal)

    return data_loaders


def get_data_dict(dataloader, iterator):
    try:
        data_dict, data_name = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        data_dict, data_name = next(iterator)
    del data_name
    return data_dict


def ramp_up(weight, ratio, step, total_steps, starting=10):
    '''
    Args:
        weight: final target weight value
        ratio: ratio between the length of ramping up and the total steps
        step: current step
        total_steps: total steps
        starting: starting step for ramping up
    Returns:
        current weight value
    '''
    # For the 1st 50 steps, the weighting is zero
    # For the ramp-up stage from starting through the length of ramping up, we linearly gradually ramp up the weight
    ramp_up_length = int(ratio*total_steps)
    if step < starting:
        return 0.0
    elif step < (ramp_up_length+starting):
        current_weight = weight * (step-starting) / ramp_up_length
        return min(current_weight, weight)
    else:
        return weight








