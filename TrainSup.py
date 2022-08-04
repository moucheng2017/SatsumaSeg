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
from Models2D import Unet
from libs.Utils import train_base

# training options control panel:
from TrainArguments import parser

# training data loader:
from libs.DataloaderOrthogonal import getData


def main():
    global args
    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for j in range(1, args.repeat + 1):
        # commented out for wandb library:
        # wandb.init(project="test-project", entity="satsuma")
        # wandb.config = {
        #     "learning_rate": learning_rate,
        #     "epochs": num_steps,
        #     "batch_size": train_batchsize
        # }

        repeat_str = str(j)
        Exp = Unet(in_ch=args.input_dim,
                   width=args.width,
                   depth=args.depth,
                   classes=args.output_dim,
                   norm='in',
                   side_output=False)

        Exp_name = 'OGnet2D' # Orthogonal Net 2D
        Exp_name = Exp_name + \
                   '_e' + str(repeat_str) + \
                   '_l' + str(args.lr) + \
                   '_b' + str(args.batch) + \
                   '_w' + str(args.width) + \
                   '_d' + str(args.depth) + \
                   '_i' + str(args.iterations) + \
                   '_l2_' + str(args.l2) + \
                   '_c_' + str(args.contrast) + \
                   '_n_' + str(args.norm) + \
                   '_t' + str(args.temp) + \
                   '_m' + str(args.lung_mask)

        data = getData(data_directory=args.data_path,
                       dataset_name=args.dataset,
                       train_batchsize=args.batch,
                       norm=args.norm,
                       contrast_aug=args.contrast,
                       lung_window=True,
                       resolution=args.resolution,
                       train_full=True,
                       unlabelled=False)

        trainSingleModel(model=Exp,
                         model_name=Exp_name,
                         num_iterations=args.iterations,
                         learning_rate=args.lr,
                         dataset_name=args.dataset,
                         apply_lung_mask=args.lung_mask,
                         trainloader_with_labels=data.get('train_loader_l'),
                         log_tag=args.log_tag,
                         l2=args.l2,
                         temp=args.temp,
                         resume=args.resume,
                         last_model=args.checkpoint_path,
                         save_iteration_starting=args.saving_starting,
                         save_iteration_interval=args.saving_interval,
                         output_dim=args.output_dim,
                         )


def trainSingleModel(model,
                     model_name,
                     num_iterations,
                     learning_rate,
                     output_dim,
                     dataset_name,
                     apply_lung_mask,
                     trainloader_with_labels,
                     temp,
                     log_tag,
                     l2,
                     save_iteration_starting,
                     save_iteration_interval,
                     resume=False,
                     last_model='/path/to/checkpoint'):

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + log_tag
    Path(saved_information_path).mkdir(parents=True, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    Path(saved_log_path).mkdir(parents=True, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    Path(saved_model_path).mkdir(parents=True, exist_ok=True)

    if output_dim == 1:
        single_channel_output = True
    else:
        single_channel_output = False

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    # resume training:
    if resume is True:
        model = torch.load(last_model)
        # checkpoint = torch.load(last_model)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch_current = checkpoint['epoch']
        # loss_current = checkpoint['loss']

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)

    start = timeit.default_timer()
    iterator_train_labelled = iter(trainloader_with_labels)

    train_mean_iu_d_tracker = 0.0
    train_mean_iu_h_tracker = 0.0
    train_mean_iu_w_tracker = 0.0

    for step in range(num_iterations):

        model.train()

        try:
            labelled_dict, labelled_name = next(iterator_train_labelled)
        except StopIteration:
            iterator_train_labelled = iter(trainloader_with_labels)
            labelled_dict, labelled_name = next(iterator_train_labelled)

        loss_d, train_mean_iu_d_ = train_base(labelled_dict["plane_d"][0], labelled_dict["plane_d"][1], labelled_dict["plane_d"][2], device, model, temp, apply_lung_mask, single_channel_output)
        loss_h, train_mean_iu_h_ = train_base(labelled_dict["plane_h"][0], labelled_dict["plane_h"][1], labelled_dict["plane_h"][2], device, model, temp, apply_lung_mask, single_channel_output)
        loss_w, train_mean_iu_w_ = train_base(labelled_dict["plane_w"][0], labelled_dict["plane_w"][1], labelled_dict["plane_w"][2], device, model, temp, apply_lung_mask, single_channel_output)
        loss = loss_w + loss_d + loss_h

        del labelled_dict
        del labelled_name

        if loss != 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * ((1 - float(step) / num_iterations) ** 0.99)

        if math.isnan(train_mean_iu_d_) is True or train_mean_iu_d_ == 0.0:
            train_mean_iu_d_tracker, train_iou_d = train_mean_iu_d_tracker, train_mean_iu_d_tracker
        else:
            train_mean_iu_d_tracker, train_iou_d = 0.9*train_mean_iu_d_tracker + 0.1*train_mean_iu_d_, train_mean_iu_d_

        if math.isnan(train_mean_iu_h_) is True or train_mean_iu_h_ == 0.0:
            train_mean_iu_h_tracker, train_iou_h = train_mean_iu_h_tracker, train_mean_iu_h_tracker
        else:
            train_mean_iu_h_tracker, train_iou_h = 0.9*train_mean_iu_h_tracker + 0.1*train_mean_iu_h_, train_mean_iu_h_

        if math.isnan(train_mean_iu_w_) is True or train_mean_iu_w_ == 0.0:
            train_mean_iu_w_tracker, train_iou_w = train_mean_iu_w_tracker, train_mean_iu_w_tracker
        else:
            train_mean_iu_w_tracker, train_iou_w = 0.9*train_mean_iu_w_tracker + 0.1*train_mean_iu_w_, train_mean_iu_w_

        print(
            'Step [{}/{}], '
            'lr: {:.4f},'
            'Train iou d: {:.4f}, '
            'Train iou h: {:.4f}, '
            'Train iou w: {:.4f}, '.format(step + 1, num_iterations,
                                           optimizer.param_groups[0]["lr"],
                                           train_iou_d,
                                           train_iou_h,
                                           train_iou_w))

        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #

        writer.add_scalars('acc metrics', {'train iou d': train_iou_d,
                                           'train iou h': train_iou_h,
                                           'train iou w': train_iou_w}, step + 1)
        # wandb.log({"loss": loss,
        #            "val iou": {
        #            "d": validate_iou_d,
        #            "h": validate_iou_h,
        #            "w": validate_iou_w},
        #            "train iou": {
        #            "d": train_iou_d,
        #            "h": train_iou_h,
        #            "w": train_iou_w}
        #            })
        # wandb.watch(model)

        if step > save_iteration_starting and step % save_iteration_interval == 0:
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)
            # torch.save({'epoch': step,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'loss': loss}, path_model)

        elif step > num_iterations - 50 and step % 2 == 0:
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

    save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(num_iterations) + '.pt'
    torch.save(model, save_model_name_full)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    save_path = saved_information_path + '/results'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
