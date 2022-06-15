import os
import torch
# torch.manual_seed(0)
# # torch.backends.cudnn.benchmark = False
import timeit
import torch.nn as nn
import numpy as np
from torch.utils import data
import shutil
import torch.nn.functional as F

from Metrics import segmentation_scores
from dataloaders.DataloaderOrthogonalNoPadding import CT_Dataset_Orthogonal_Fast
from tensorboardX import SummaryWriter

# import wandb

from Utils import validate_three_planes
from Loss import SoftDiceLoss
# ==============================================
from Models2DOrthogonal import Unet2DMultiChannel
import errno

from Utils import train_base

from analysis.Inference3D import test_all_models

# This script trains a weird model:
# We train on three planes simulatenously


def trainModels(dataset_name,
                data_directory,
                repeat,
                num_steps,
                learning_rate,
                width,
                log_tag,
                train_batchsize,
                val_batchsize=3,
                new_d=5,
                new_h=480,
                new_w=480,
                new_z=320,
                temp=0.5,
                l2=0.01
                ):

    for j in range(1, repeat + 1):
        # wandb.init(project="test-project", entity="satsuma")
        # wandb.config = {
        #     "learning_rate": learning_rate,
        #     "epochs": num_steps,
        #     "batch_size": train_batchsize
        # }
        repeat_str = str(j)
        Exp = Unet2DMultiChannel(in_ch=new_d, width=width, output_channels=new_d)
        Exp_name = 'OrthogonalSup2DFast'

        Exp_name = Exp_name + \
                   '_e' + str(repeat_str) + \
                   '_l' + str(learning_rate) + \
                   '_b' + str(train_batchsize) + \
                   '_w' + str(width) + \
                   '_s' + str(num_steps) + \
                   '_r' + str(l2) + \
                   '_d' + str(new_d) + \
                   '_h' + str(new_h) + \
                   '_w' + str(new_w) + \
                   '_z' + str(new_z) + \
                   '_t' + str(temp)

        trainloader_withlabels, validateloader, test_data_path, train_dataset_with_labels, validate_dataset = getData(data_directory,
                                                                                                                      dataset_name,
                                                                                                                      train_batchsize,
                                                                                                                      [new_d, new_h, new_w],
                                                                                                                      [new_z, new_d, new_w],
                                                                                                                      [new_z, new_w, new_d],
                                                                                                                      val_batchsize)

        trainSingleModel(model=Exp,
                         model_name=Exp_name,
                         num_steps=num_steps,
                         learning_rate=learning_rate,
                         dataset_name=dataset_name,
                         trainloader_with_labels=trainloader_withlabels,
                         validateloader=validateloader,
                         log_tag=log_tag,
                         l2=l2,
                         temp=temp)


def getData(data_directory, dataset_name, train_batchsize, d, h, w, val_batchsize=5):

    data_directory = data_directory + '/' + dataset_name

    folder_labelled = data_directory + '/labelled'

    train_image_folder_labelled = folder_labelled + '/imgs'
    train_label_folder_labelled = folder_labelled + '/lbls'
    train_lung_folder_labelled = folder_labelled + '/lung'

    train_dataset_labelled = CT_Dataset_Orthogonal_Fast(train_image_folder_labelled, train_label_folder_labelled, train_lung_folder_labelled, d, h, w, labelled=True)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=True)

    validate_image_folder = data_directory + '/validate/imgs'
    validate_label_folder = data_directory + '/validate/lbls'
    validate_lung_folder = data_directory + '/validate/lung'

    validate_dataset = CT_Dataset_Orthogonal_Fast(validate_image_folder, validate_label_folder, validate_lung_folder, d, h, w, labelled=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=val_batchsize, shuffle=True, num_workers=0, drop_last=True)

    testdata_path = data_directory + '/test'

    return trainloader_labelled, validateloader, testdata_path, train_dataset_labelled, validate_dataset
# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_steps,
                     learning_rate,
                     dataset_name,
                     trainloader_with_labels,
                     validateloader,
                     temp,
                     log_tag,
                     l2,
                     resume=False,
                     last_model='/path/to/checkpoint'):

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + log_tag
    if not os.path.exists(saved_information_path):
        os.makedirs(saved_information_path, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    if not os.path.exists(saved_log_path):
        os.makedirs(saved_log_path, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path, exist_ok=True)

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    model.to(device)

    # resume training:
    if resume:
        model = torch.load(last_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)

    start = timeit.default_timer()

    iterator_train_labelled = iter(trainloader_with_labels)

    for step in range(num_steps):

        model.train()
        train_iou_d = []
        train_iou_h = []
        train_iou_w = []

        try:
            labelled_dict, labelled_name = next(iterator_train_labelled)
        except StopIteration:
            iterator_train_labelled = iter(trainloader_with_labels)
            labelled_dict, labelled_name = next(iterator_train_labelled)

        loss_d, train_mean_iu_d_ = train_base(labelled_dict["plane_d"][0], labelled_dict["plane_d"][1], labelled_dict["plane_d"][2], device, model, temp)
        loss_h, train_mean_iu_h_ = train_base(labelled_dict["plane_h"][0], labelled_dict["plane_h"][1], labelled_dict["plane_h"][2], device, model, temp)
        loss_w, train_mean_iu_w_ = train_base(labelled_dict["plane_w"][0], labelled_dict["plane_w"][1], labelled_dict["plane_w"][2], device, model, temp)
        loss = loss_w + loss_d + loss_h
        del labelled_dict
        del labelled_name
        train_iou_d.append(train_mean_iu_d_)
        train_iou_h.append(train_mean_iu_h_)
        train_iou_w.append(train_mean_iu_w_)
        train_iou_d = np.nanmean(train_iou_d)
        train_iou_h = np.nanmean(train_iou_h)
        train_iou_w = np.nanmean(train_iou_w)

        validate_ious = validate_three_planes(validateloader, device, model)
        validate_iou_d = np.nanmean(validate_ious["val d plane"])
        validate_iou_h = np.nanmean(validate_ious["val h plane"])
        validate_iou_w = np.nanmean(validate_ious["val w plane"])
        del validate_ious

        if loss != 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)

        print(
            'Step [{}/{}], '
            'lr: {:.4f},'
            'Train iou d: {:.4f}, '
            'Train iou h: {:.4f}, '
            'Train iou w: {:.4f}, '
            'val iou d:{:.4f}, '
            'val iou h:{:.4f}, '
            'val iou w:{:.4f}, '.format(step + 1, num_steps,
                                      optimizer.param_groups[0]["lr"],
                                      train_iou_d,
                                      train_iou_h,
                                      train_iou_w,
                                      validate_iou_d,
                                      validate_iou_h,
                                      validate_iou_w))


        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #

        writer.add_scalars('acc metrics', {'train iou d': train_iou_d,
                                           'train iou h': train_iou_h,
                                           'train iou w': train_iou_w,
                                           'val iou d': validate_iou_d,
                                           'val iou h': validate_iou_h,
                                           'val iou w': validate_iou_w}, step + 1)

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

        # # if step > num_steps - 20:
        # if step > num_steps - 100:
        #     save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
        #     path_model = save_model_name_full
        #     torch.save(model, path_model)

        if step > 2000 and step % 50 == 0:
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
            torch.save(model, path_model)

    # save_model_name_full = saved_model_path + '/' + save_model_name + '.pt'
    # path_model = save_model_name_full
    # torch.save(model, path_model)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    save_path = saved_information_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # iou_mean, iou_std = test_all_models(saved_model_path, testdata_path, save_path, size, class_no, False, False)
    #
    # print('Test IoU: ' + str(iou_mean) + '\n')
    # print('Test IoU std: ' + str(iou_std) + '\n')

    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
