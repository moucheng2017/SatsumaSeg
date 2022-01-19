import os
import errno
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

from Utils import evaluate, test
from Loss import SoftDiceLoss
# =================================
from Baselines import Unet3D


# For consistency restrictions:
# 1. Global and local, same resolution
# 2. High resolution and low resolution


class GlobalLocal(object):
    def __init__(self, global_local_ratios=[0.5, 0.5, 0.5]):
        self.global_local_ratios = global_local_ratios

    def crop(self, global_input, global_seg):
        b, c, d, h, w = global_input.size()
        new_d = int(d * self.global_local_ratios[0])
        new_h = int(h * self.global_local_ratios[1])
        new_w = int(w * self.global_local_ratios[2])

        if new_h < h:
            top_h = np.random.randint(0, h - new_h)
        else:
            top_h = 0

        if new_w < w:
            top_w = np.random.randint(0, w - new_w)
        else:
            top_w = 0

        if new_d < d:
            top_d = np.random.randint(0, d - new_d)
        else:
            top_d = 0

        local_input = global_input[
                      :,
                      :,
                      top_d:top_d + new_d,
                      top_h:top_h + new_h,
                      top_w:top_w + new_w
                      ]

        local_seg = global_seg[
                      :,
                      :,
                      top_d:top_d + new_d,
                      top_h:top_h + new_h,
                      top_w:top_w + new_w
                      ]

        return local_input, local_seg


class HighLow(object):
    def __init__(self, high_low_ratios=[2, 2, 2]):
        self.high_low_ratios = high_low_ratios

    def crop(self, global_input, global_seg):
        local_input = global_input[
                      :,
                      :,
                      ::self.high_low_ratios[0],
                      ::self.high_low_ratios[1],
                      ::self.high_low_ratios[2]
                      ]

        local_seg = global_seg[
                      :,
                      :,
                      ::self.high_low_ratios[0],
                      ::self.high_low_ratios[1],
                      ::self.high_low_ratios[2]
                      ]

        return local_input, local_seg


def trainModels(dataset_tag,
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
                new_resolution=64,
                # lr_decay='poly',
                spatial_consistency='global_local'
                ):

    for j in range(1, repeat + 1):

        repeat_str = str(j)

        Exp = Unet3D(in_ch=input_dim, width=width, class_no=class_no, z_downsample=downsample)
        Exp_name = 'sup_unet3d' + \
                   '_e_' + str(repeat_str) + \
                   '_l' + str(learning_rate) + \
                   '_b' + str(train_batchsize) + \
                   '_w' + str(width) + \
                   '_s' + str(num_steps) + \
                   '_d' + str(downsample) + \
                   '_r' + str(new_resolution) + \
                   '_restriction_' + str(spatial_consistency)

        trainloader_withlabels, validateloader, test_data_path, train_dataset_with_labels, validate_dataset, test_dataset = getData(data_directory, dataset_name, dataset_tag, train_batchsize, new_resolution)

        # ===================
        trainSingleModel(model=Exp,
                         model_name=Exp_name,
                         num_steps=num_steps,
                         learning_rate=learning_rate,
                         dataset_name=dataset_name,
                         dataset_tag=dataset_tag,
                         train_dataset_with_labels=train_dataset_with_labels,
                         train_batchsize=train_batchsize,
                         trainloader_with_labels=trainloader_withlabels,
                         validateloader=validateloader,
                         testdata_path=test_data_path,
                         class_no=class_no,
                         log_tag=log_tag,
                         dilation=1,
                         # lr_decay=lr_decay,
                         spatial_consistency=spatial_consistency)


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, new_resolution):

    data_directory = data_directory + dataset_name + '/' + dataset_tag
    data_directory_eval_test = data_directory + dataset_name

    folder_labelled = data_directory + '/labelled'

    train_image_folder_labelled = folder_labelled + '/patches'
    train_label_folder_labelled = folder_labelled + '/labels'
    train_dataset_labelled = CT_Dataset(train_image_folder_labelled, train_label_folder_labelled, new_resolution, labelled=True)

    # train_image_folder_unlabelled = data_directory + '/unlabelled/patches'
    # train_label_folder_unlabelled = data_directory + '/unlabelled/labels'
    # train_dataset_unlabelled = CustomDataset(train_image_folder_unlabelled, train_label_folder_unlabelled, 'none', labelled=True)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=True)
    # trainloader_unlabelled = data.DataLoader(train_dataset_unlabelled, batch_size=train_batchsize*ratio, shuffle=True, num_workers=0, drop_last=False)

    validate_image_folder = data_directory + '/validate/patches'
    validate_label_folder = data_directory + '/validate/labels'

    testdata_path = data_directory + '/test'

    test_image_folder = data_directory + '/test/patches'
    test_label_folder = data_directory + '/test/labels'

    validate_dataset = CT_Dataset(validate_image_folder, validate_label_folder, new_resolution, labelled=True)
    test_dataset = CT_Dataset(test_image_folder, test_label_folder, new_resolution, labelled=True)

    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    return trainloader_labelled, validateloader, testdata_path, train_dataset_labelled, validate_dataset, test_dataset
# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_steps,
                     learning_rate,
                     dataset_name,
                     dataset_tag,
                     train_dataset_with_labels,
                     train_batchsize,
                     trainloader_with_labels,
                     validateloader,
                     dilation,
                     testdata_path,
                     log_tag,
                     class_no,
                     # lr_decay,
                     spatial_consistency):

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + dataset_tag + '/' + log_tag
    if not os.path.exists(saved_information_path):
        os.makedirs(saved_information_path)
    # os.mkdir(saved_information_path, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    # os.mkdir(saved_log_path, exist_ok=True)
    if not os.path.exists(saved_log_path):
        os.makedirs(saved_log_path)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    # os.mkdir(saved_model_path, exist_ok=True)
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-4)

    start = timeit.default_timer()

    iterator_train_labelled = iter(trainloader_with_labels)

    for step in range(num_steps):

        model.train()
        train_iou = []
        train_sup_loss = []

        try:
            labelled_img, labelled_label, labelled_name = next(iterator_train_labelled)
        except StopIteration:
            iterator_train_labelled = iter(trainloader_with_labels)
            labelled_img, labelled_label, labelled_name = next(iterator_train_labelled)

        train_imgs = labelled_img.to(device=device, dtype=torch.float32)
        labels = labelled_label.to(device=device, dtype=torch.float32)

        outputs = model(train_imgs, [dilation, dilation, dilation, dilation], [dilation, dilation, dilation, dilation])
        if class_no == 2:
            prob_outputs = torch.sigmoid(outputs)
        else:
            prob_outputs = F.softmax(outputs, dim=1)

        if class_no == 2:
            loss = SoftDiceLoss()(prob_outputs, labels)
        else:
            loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=8)(prob_outputs, labels.long().squeeze(1))

        train_sup_loss.append(loss.item())

        if class_no == 2:
            class_outputs = (prob_outputs > 0.5).float()
        else:
            _, class_outputs = torch.max(prob_outputs, dim=1)

        train_mean_iu_ = segmentation_scores(labels, class_outputs, class_no)
        train_iou.append(train_mean_iu_)

        validate_iou, validate_h_dist = evaluate(validateloader, model, device, model_name, class_no, dilation)

        # Adding Spatial Constraints:
        regularisation = 0.0
        if spatial_consistency == 'global_local':
            Cropping = GlobalLocal([0.5, 0.5, 0.5])
            # Cropping = GlobalLocal([random.uniform(0.4, 0.8), random.uniform(0.4, 0.8), random.uniform(0.4, 0.8)])
            local_train_img, local_seg = Cropping.crop(train_imgs, class_outputs)
            outputs2 = model(local_train_img.detach(), [dilation, dilation, dilation, dilation], [dilation, dilation, dilation, dilation])
            if class_no == 2:
                prob_outputs2 = torch.sigmoid(outputs2)
                outputs2 = (prob_outputs2 > 0.5).float()
            else:
                prob_outputs2 = F.softmax(outputs2, dim=1)
                _, outputs2 = torch.max(prob_outputs2, dim=1)
            regularisation = nn.MSELoss(reduction='mean')(outputs2, local_seg)
            loss += 0.01 * regularisation

        elif spatial_consistency == 'high_low':
            # Cropping = HighLow([random.choice([1, 2]), random.choice([1, 2]), random.choice([1, 2])])
            Cropping = HighLow([2, 2, 2])
            local_train_img, local_seg = Cropping.crop(train_imgs, class_outputs)
            outputs2 = model(local_train_img.detach(), [dilation, dilation, dilation, dilation], [dilation, dilation, dilation, dilation])
            if class_no == 2:
                prob_outputs2 = torch.sigmoid(outputs2)
                outputs2 = (prob_outputs2 > 0.5).float()
            else:
                prob_outputs2 = F.softmax(outputs2, dim=1)
                _, outputs2 = torch.max(prob_outputs2, dim=1)
            regularisation = nn.MSELoss(reduction='mean')(outputs2, local_seg)
            loss += 0.01*regularisation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if lr_decay == 'poly':
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)
        # elif lr_decay == 'steps:':
        #     for param_group in optimizer.param_groups:
        #         if step % (num_steps // 4) == 0:
        #             param_group["lr"] = param_group["lr"] * 0.5
        # elif lr_decay == 'warmup':
        #     warm_up_lr = 0.8
        #     for param_group in optimizer.param_groups:
        #         if step < num_steps*warm_up_lr:
        #             param_group["lr"] = step * learning_rate / (num_steps*warm_up_lr)
        #         else:
        #             param_group["lr"] = learning_rate
        # else:
        #     pass

        print(
            'Step [{}/{}], '
            'lr: {:.4f},'
            'Train sup loss: {:.4f}, '
            'Train iou: {:.4f}, '
            'Train regularisation: {:.4f}, '
            'val iou:{:.4f}, '.format(step + 1, num_steps,
                                      optimizer.param_groups[0]["lr"],
                                      np.nanmean(train_sup_loss),
                                      np.nanmean(train_iou),
                                      regularisation,
                                      np.nanmean(validate_iou)))

        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #

        writer.add_scalars('acc metrics', {'train iou': np.nanmean(train_iou),
                                           'val hausdorff dist': np.nanmean(validate_h_dist),
                                           'val iou': np.nanmean(validate_iou)}, step + 1)

        writer.add_scalars('loss values', {'sup loss': np.nanmean(train_sup_loss)}, step + 1)

        if step > num_steps - 10:
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            path_model = save_model_name_full
            torch.save(model, path_model)

    # save_model_name_full = saved_model_path + '/' + save_model_name + '.pt'
    # path_model = save_model_name_full
    # torch.save(model, path_model)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    test_image_path = os.path.join(testdata_path, 'patches')
    test_label_path = os.path.join(testdata_path, 'labels')
    test_iou = test(saved_information_path + '/' + save_model_name,
                    saved_model_path,
                    test_image_path,
                    test_label_path,
                    device,
                    model_name,
                    class_no,
                    [192, 192, 192],
                    1)

    print('Test IoU: ' + str(np.nanmean(test_iou)) + '\n')
    print('Test IoU std: ' + str(np.nanstd(test_iou)) + '\n')
    # print('Test H-dist: ' + str(np.nanmean(test_h_dist)) + '\n')
    # print('Test H-dist std: ' + str(np.nanstd(test_h_dist)) + '\n')
    # print('Test recall: ' + str(np.nanmean(test_recall)) + '\n')
    # print('Test recall std: ' + str(np.nanstd(test_recall)) + '\n')
    # print('Test precision: ' + str(np.nanmean(test_precision)) + '\n')
    # print('Test precision std: ' + str(np.nanstd(test_precision)) + '\n')

    print('\nTraining finished and model saved\n')
    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
