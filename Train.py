# basic libs:
import math
import torch
import timeit
import shutil
from pathlib import Path


# Tracking the training process:
# import wandb
from tensorboardX import SummaryWriter

# model:
from Models2D import Unet, UnetBPL
from libs.Utils import train_base

# training options control panel:
from Arguments import parser

# training data loader:
from libs.DataloaderOrthogonal import getData
from libs import TrainBase
from libs import Helpers


def main():
    global args
    args = parser.parse_args()

    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)

    # resume training:
    if args.resume is True:
        model = torch.load(args.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2)

    # data loaders:
    data_loaders = Helpers.get_data_simple_wrapper(args)

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start_time = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(data_loaders, args)

    # running loop:
    # for step in range(args.iterations):











