import math
import os
import torch
# torch.manual_seed(0)
# # torch.backends.cudnn.benchmark = False
import timeit
from torch.utils import data
import shutil

from pathlib import Path

from dataloaders.DataloaderOrthogonal import CT_Dataset_Orthogonal
from tensorboardX import SummaryWriter

from training_arguments_sup import parser
# import wandb

# ==============================================
from Models2D import Unet
import errno

from Utils import train_base


# This script trains a weird model on three planes simulatenously


def trainModels():

    global args
    args = parser.parse_args()

    for j in range(1, repeat + 1):
        # wandb.init(project="test-project", entity="satsuma")
        # wandb.config = {
        #     "learning_rate": learning_rate,
        #     "epochs": num_steps,
        #     "batch_size": train_batchsize
        # }

        repeat_str = str(j)
        Exp = Unet(in_ch=1, width=width, depth=depth, classes=1, norm='in', side_output=False)
        Exp_name = 'OrthogonalSup2DSingle'

        Exp_name = Exp_name + \
                   '_e' + str(repeat_str) + \
                   '_l' + str(learning_rate) + \
                   '_b' + str(train_batchsize) + \
                   '_w' + str(width) + \
                   '_d' + str(depth) + \
                   '_s' + str(num_steps) + \
                   '_r' + str(l2) + \
                   '_c_' + str(contrast) + \
                   '_n_' + str(norm) + \
                   '_t' + str(temp)

        trainloader_withlabels = getData(data_directory, dataset_name, train_batchsize, norm, contrast, lung)

        trainSingleModel(model=Exp,
                         model_name=Exp_name,
                         num_steps=num_steps,
                         learning_rate=learning_rate,
                         dataset_name=dataset_name,
                         trainloader_with_labels=trainloader_withlabels,
                         log_tag=log_tag,
                         l2=l2,
                         temp=temp,
                         resume=resume_training,
                         last_model=checkpoint_path)


def getData(data_directory, dataset_name, train_batchsize, norm=False, contrast_aug=False, lung_window=True, apply_lung_mask=True):

    data_directory = data_directory + '/' + dataset_name

    folder_labelled = data_directory + '/labelled'

    train_image_folder_labelled = folder_labelled + '/imgs'
    train_label_folder_labelled = folder_labelled + '/lbls'
    train_lung_folder_labelled = folder_labelled + '/lung'

    train_dataset_labelled = CT_Dataset_Orthogonal(imgs_folder=train_image_folder_labelled,
                                                   labels_folder=train_label_folder_labelled,
                                                   lung_folder=train_lung_folder_labelled,
                                                   labelled=True,
                                                   full_resolution=512,
                                                   normalisation=norm,
                                                   contrast_aug=contrast_aug,
                                                   lung_window=lung_window)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=True)

    # validate_image_folder = data_directory + '/validate/imgs'
    # validate_label_folder = data_directory + '/validate/lbls'
    # validate_lung_folder = data_directory + '/validate/lung'
    #
    # validate_dataset = CT_Dataset_Orthogonal(validate_image_folder, validate_label_folder, validate_lung_folder, d, h, w, labelled=True)
    # validateloader = data.DataLoader(validate_dataset, batch_size=val_batchsize, shuffle=True, num_workers=0, drop_last=True)
    #
    # testdata_path = data_directory + '/test'

    # return trainloader_labelled, validateloader, testdata_path, train_dataset_labelled, validate_dataset
    return trainloader_labelled
# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_steps,
                     learning_rate,
                     dataset_name,
                     trainloader_with_labels,
                     temp,
                     log_tag,
                     l2,
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

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)

    # resume training:
    if resume is True:
        checkpoint = torch.load(last_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_current = checkpoint['epoch']
        loss_current = checkpoint['loss']

    start = timeit.default_timer()
    iterator_train_labelled = iter(trainloader_with_labels)

    train_mean_iu_d_tracker = 0.0
    train_mean_iu_h_tracker = 0.0
    train_mean_iu_w_tracker = 0.0

    for step in range(num_steps):

        model.train()

        try:
            labelled_dict, labelled_name = next(iterator_train_labelled)
        except StopIteration:
            iterator_train_labelled = iter(trainloader_with_labels)
            labelled_dict, labelled_name = next(iterator_train_labelled)

        loss_d, train_mean_iu_d_ = train_base(labelled_dict["plane_d"][0], labelled_dict["plane_d"][1], labelled_dict["plane_d"][2], device, model, temp, False, True)
        loss_h, train_mean_iu_h_ = train_base(labelled_dict["plane_h"][0], labelled_dict["plane_h"][1], labelled_dict["plane_h"][2], device, model, temp, False, True)
        loss_w, train_mean_iu_w_ = train_base(labelled_dict["plane_w"][0], labelled_dict["plane_w"][1], labelled_dict["plane_w"][2], device, model, temp, False, True)
        loss = loss_w + loss_d + loss_h
        del labelled_dict
        del labelled_name

        if loss != 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)

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
            'Train iou w: {:.4f}, '.format(step + 1, num_steps,
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
        #
        # wandb.watch(model)

        # if step > 2000 and step % 100 == 0:
        if step % 10 == 0:
            # save checker points
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            path_model = save_model_name_full
            # torch.save(model, path_model)
            torch.save({'epoch': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, path_model)

        elif step > num_steps - 50:
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            path_model = save_model_name_full
            torch.save({'epoch': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, path_model)

    # save_model_name_full = saved_model_path + '/' + save_model_name + '.pt'
    # path_model = save_model_name_full
    # torch.save(model, path_model)

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