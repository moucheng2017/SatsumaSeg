import os
import torch
# torch.manual_seed(0)
# # torch.backends.cudnn.benchmark = False
import timeit
import torch.nn as nn
import random
import numpy as np
from torch.utils import data
import shutil
import torch.nn.functional as F

from Metrics import segmentation_scores
from dataloaders.Dataloader import CT_Dataset
from tensorboardX import SummaryWriter

from Utils import evaluate, test, sigmoid_rampup
from Loss import SoftDiceLoss
# =================================
from Models import Unet3D
from analysis.VolumeSegmentation import test_all_models
import errno


def trainModels(
                dataset_name,
                data_directory,
                downsample,
                input_dim,
                class_no,
                repeat,
                train_batchsize,
                num_steps,
                learning_rate,
                width,
                log_tag,
                unlabelled=5,
                new_resolution=[12, 512, 512],
                l2=0.01,
                alpha=1.0,
                warmup=0.1,
                threshold=0.5
                ):

    for j in range(1, repeat + 1):

        assert threshold > 0.0

        repeat_str = str(j)

        Exp = Unet3D(in_ch=input_dim, width=width, class_no=class_no, z_downsample=downsample)

        if threshold < 1.0:
            Exp_name = 'SegPL' + \
                       '_e' + str(repeat_str) + \
                       '_l' + str(learning_rate) + \
                       '_t' + str(threshold) + \
                       '_b' + str(train_batchsize) + \
                       '_u' + str(unlabelled) + \
                       '_w' + str(width) + \
                       '_s' + str(num_steps) + \
                       '_d' + str(downsample) + \
                       '_r' + str(l2) + \
                       '_a' + str(alpha) + \
                       '_wu' + str(warmup) + \
                       '_z' + str(new_resolution[0]) + \
                       '_x' + str(new_resolution[1])
        else:
            # random threshld between 0 and 1 which is sampled from half of the normal distribution
            Exp_name = 'SegPL' + \
                       '_e' + str(repeat_str) + \
                       '_l' + str(learning_rate) + \
                       '_t_r' + \
                       '_b' + str(train_batchsize) + \
                       '_u' + str(unlabelled) + \
                       '_w' + str(width) + \
                       '_s' + str(num_steps) + \
                       '_d' + str(downsample) + \
                       '_r' + str(l2) + \
                       '_a' + str(alpha) + \
                       '_wu' + str(warmup) + \
                       '_z' + str(new_resolution[0]) + \
                       '_x' + str(new_resolution[1])

        trainloader_withlabels, trainloader_withoutlabels, validateloader = getData(data_directory,
                                                                                    dataset_name,
                                                                                    train_batchsize,
                                                                                    new_resolution,
                                                                                    unlabelled)

        # ===================
        trainSingleModel(model=Exp,
                         model_name=Exp_name,
                         num_steps=num_steps,
                         learning_rate=learning_rate,
                         dataset_name=dataset_name,
                         trainloader_with_labels=trainloader_withlabels,
                         trainloader_without_labels=trainloader_withoutlabels,
                         validateloader=validateloader,
                         class_no=class_no,
                         log_tag=log_tag,
                         dilation=1,
                         l2=l2,
                         alpha=alpha,
                         warmup=warmup,
                         threshold=threshold
                         )


def getData(data_directory, dataset_name, train_batchsize, new_resolution, ratio):

    data_directory = data_directory + '/' + dataset_name

    folder_labelled = data_directory + '/labelled'

    train_image_folder_labelled = folder_labelled + '/imgs'
    train_label_folder_labelled = folder_labelled + '/lbls'
    train_lung_folder_labelled = folder_labelled + '/lung'

    train_dataset_labelled = CT_Dataset(train_image_folder_labelled, train_label_folder_labelled, train_lung_folder_labelled, new_resolution, labelled=True)

    train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'
    train_label_folder_unlabelled = data_directory + '/unlabelled/lbls'
    train_lung_folder_unlabelled = data_directory + '/unlabelled/lung'
    train_dataset_unlabelled = CT_Dataset(train_image_folder_unlabelled, train_label_folder_unlabelled, train_lung_folder_unlabelled, new_resolution, labelled=False)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=True)
    trainloader_unlabelled = data.DataLoader(train_dataset_unlabelled, batch_size=train_batchsize*ratio, shuffle=True, num_workers=0, drop_last=False)

    validate_image_folder = data_directory + '/validate/imgs'
    validate_label_folder = data_directory + '/validate/lbls'
    validate_lung_folder = data_directory + '/validate/lung'

    validate_dataset = CT_Dataset(validate_image_folder, validate_label_folder, validate_lung_folder, new_resolution, labelled=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # testdata_path = data_directory + '/test'
    # test_image_folder = data_directory + '/test/imgs'
    # test_label_folder = data_directory + '/test/lbls'
    # test_lung_folder = data_directory + '/test/lung'
    # test_dataset = CT_Dataset(test_image_folder, test_label_folder, test_lung_folder, new_resolution, labelled=True)
    # testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    return trainloader_labelled, trainloader_unlabelled, validateloader


def trainSingleModel(model,
                     model_name,
                     num_steps,
                     learning_rate,
                     dataset_name,
                     trainloader_with_labels,
                     trainloader_without_labels,
                     validateloader,
                     dilation,
                     log_tag,
                     class_no,
                     l2=0.01,
                     alpha=1.0,
                     warmup=0.1,
                     threshold=0.5):

    # alpha = 1.0

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + '/' + log_tag

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)

    start = timeit.default_timer()

    iterator_train_labelled = iter(trainloader_with_labels)
    iterator_train_unlabelled = iter(trainloader_without_labels)

    for step in range(num_steps):

        model.train()
        train_iou = []
        train_sup_loss = []
        train_unsup_loss = []

        if step < int(warmup * num_steps):
            scale = sigmoid_rampup(step, int(warmup * num_steps), 1.0)
            alpha_current = alpha * scale
        else:
            alpha_current = alpha

        try:
            labelled_img, labelled_label, labelled_lung, labelled_name = next(iterator_train_labelled)
            unlabelled_img, unlabelled_lung, unlabelled_name = next(iterator_train_unlabelled)
        except StopIteration:
            iterator_train_labelled = iter(trainloader_with_labels)
            labelled_img, labelled_label, labelled_lung, labelled_name = next(iterator_train_labelled)

            iterator_train_unlabelled = iter(trainloader_without_labels)
            unlabelled_img, unlabelled_lung, unlabelled_name = next(iterator_train_unlabelled)

        train_imgs_l = labelled_img.to(device=device, dtype=torch.float32)
        b_l, d, c, h, w = train_imgs_l.size()

        train_imgs_u = unlabelled_img.to(device=device, dtype=torch.float32)
        b_u, d, c, h, w = train_imgs_u.size()

        # print(train_imgs_u.size())
        # print(train_imgs_l.size())

        train_imgs = torch.cat((train_imgs_l, train_imgs_u), dim=0)

        labels = labelled_label.to(device=device, dtype=torch.float32)

        labelled_lung = labelled_lung.to(device=device, dtype=torch.float32)

        unlabelled_lung = unlabelled_lung.to(device=device, dtype=torch.float32)

        if torch.sum(labels) > 10.0:

            outputs, _ = model(train_imgs, [dilation, dilation, dilation, dilation], [dilation, dilation, dilation, dilation])
            outputs, outputs_u = torch.split(outputs, [b_l, b_u], dim=0)

            prob_outputs = torch.sigmoid(outputs)

            lung_mask_labelled = (labelled_lung > 0.5)
            prob_outputs_masked = torch.masked_select(prob_outputs, lung_mask_labelled)
            labels_masked = torch.masked_select(labels, lung_mask_labelled)

            if torch.sum(prob_outputs_masked) > 10.0:
                loss_s = SoftDiceLoss()(prob_outputs_masked, labels_masked) + nn.BCELoss(reduction='mean')(prob_outputs_masked.squeeze(), labels_masked.squeeze())
            else:
                loss_s = SoftDiceLoss()(prob_outputs_masked, labels_masked)

            train_sup_loss.append(loss_s.item())

            class_outputs = (prob_outputs_masked > 0.5).float()

            train_mean_iu_ = segmentation_scores(labels_masked, class_outputs, class_no)
            train_iou.append(train_mean_iu_)

            validate_iou, validate_h_dist = evaluate(validateloader, model, device, model_name, class_no, dilation)
            prob_outputs_u = torch.sigmoid(outputs_u)

            if threshold > 1.0:
                threshold_pseudo = torch.rand(1, device=device)
            else:
                threshold_pseudo = threshold

            pseudo_label = (prob_outputs_u.detach() > threshold_pseudo).float()

            lung_mask_unlabelled = (unlabelled_lung > 0.5)
            prob_outputs_u_masked = torch.masked_select(prob_outputs_u, lung_mask_unlabelled)
            pseudo_label_masked = torch.masked_select(pseudo_label, lung_mask_unlabelled)

            # If pseudo label is all background or all foreground then we don't use it because that would be obviously wrong
            foreground_in_pseudo_labels = [torch.sum(pseudo_label_masked[i, :, :, :, :]) for i in range(pseudo_label_masked.size()[0])]
            loss_u = 0.0
            for i, foreground_index in enumerate(foreground_in_pseudo_labels):
                if 10.0 < foreground_index < torch.numel(pseudo_label_masked[0, :, :, :, :]):
                    loss_u += SoftDiceLoss()(prob_outputs_u_masked[i, :, :, :, :].copy(), pseudo_label_masked[i, :, :, :, :].copy()) + nn.BCELoss(reduction='mean')(prob_outputs_u_masked[i, :, :, :, :].copy().squeeze(), pseudo_label_masked[i, :, :, :, :].copy().squeeze())

            loss_u = loss_u * alpha_current

            if loss_u != 0:
                train_unsup_loss.append(loss_u.item())

            loss = loss_u + loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)

            print(
                'Step [{}/{}], '
                'lr: {:.4f},'
                'Train sup loss: {:.4f}, '
                'Train unsup loss: {:.4f}, '
                'Train iou: {:.4f}, '
                'val iou:{:.4f}, '.format(step + 1, num_steps,
                                          optimizer.param_groups[0]["lr"],
                                          np.nanmean(train_sup_loss),
                                          np.nanmean(train_unsup_loss),
                                          np.nanmean(train_iou),
                                          np.nanmean(validate_iou)))

            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

            writer.add_scalars('acc metrics', {'train iou': np.nanmean(train_iou),
                                               'val iou': np.nanmean(validate_iou)}, step + 1)

            writer.add_scalars('loss values', {'sup loss': np.nanmean(train_sup_loss),
                                               'unsup loss': np.nanmean(train_unsup_loss)}, step + 1)

        if step > num_steps - 5:
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            path_model = save_model_name_full
            torch.save(model, path_model)

    # save_model_name_full = saved_model_path + '/' + save_model_name + '.pt'
    # path_model = save_model_name_full
    # torch.save(model, path_model)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    save_path = saved_model_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # iou_mean, iou_std = test_all_models(saved_model_path, testdata_path, save_path, size, class_no, False, False)
    # print('Test IoU: ' + str(iou_mean) + '\n')
    # print('Test IoU std: ' + str(iou_std) + '\n')

    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
