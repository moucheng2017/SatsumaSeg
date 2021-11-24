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
# import Image
# from scipy.spatial.distance import cdist

from Metrics import segmentation_scores
from Utils import CustomDataset
from tensorboardX import SummaryWriter

from Utils import evaluate, test
from Loss import SoftDiceLoss
# =================================
from Baselines import Unet3D


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
                data_augmentation='none',
                dilation=1,
                lr_decay='poly'
                ):

    for j in range(1, repeat + 1):

        repeat_str = str(j)

        Exp = Unet3D(in_ch=input_dim, width=width, class_no=class_no, z_downsample=downsample)
        Exp_name = 'sup_unet3d' + \
                   '_r_' + str(repeat_str) + \
                   '_lr_' + str(learning_rate) + \
                   '_b_' + str(train_batchsize) + \
                   '_w_' + str(width) + \
                   '_s_' + str(num_steps) + \
                   '_di_' + str(dilation) + \
                   '_d_' + str(downsample) + \
                   '_aug_' + str(data_augmentation) + \
                   '_decay_' + str(lr_decay)

        if class_no > 2:
            multi_class = True
        else:
            multi_class = False

        if dataset_name == 'carve':
            trainloader_withlabels, validateloader, testloader, train_dataset_with_labels, validate_dataset, test_dataset = getData_carve(data_directory, dataset_name, dataset_tag, train_batchsize, data_augmentation, multi_class)
        else:
            trainloader_withlabels, validateloader, testloader, train_dataset_with_labels, validate_dataset, test_dataset = getData(data_directory, dataset_name, dataset_tag, train_batchsize, data_augmentation, multi_class)

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
                         testdata=testloader,
                         class_no=class_no,
                         log_tag=log_tag,
                         dilation=dilation,
                         lr_decay=lr_decay)


def getData_carve(data_directory, dataset_name, dataset_tag, train_batchsize, data_augmentation='none', multi_class=True):

    if multi_class is True:
        data_set_tag = '3d_multi'
    else:
        data_set_tag = '3d_binary'

    all_underline_pos = [pos for pos, char in enumerate(dataset_tag) if char == '_']

    # resolution
    resolution_imgs_start = dataset_tag.find('R')
    resolution_imgs_end = resolution_imgs_start + len('R')
    resolution_imgs = dataset_tag[resolution_imgs_end:all_underline_pos[0]]

    # case index as training
    case_start = dataset_tag.find('C')
    case_end = case_start + len('C')
    case_no = dataset_tag[case_end:all_underline_pos[1]]

    # depth
    new_depth_of_img_start = dataset_tag.find('D')
    new_depth_of_img_end = new_depth_of_img_start + len('D')
    new_depth_of_img = dataset_tag[new_depth_of_img_end:all_underline_pos[2]]

    # step/gap between each patch
    gap_between_imgs_start = dataset_tag.find('S')
    gap_between_imgs_end = gap_between_imgs_start + len('S')
    gap_between_imgs = dataset_tag[gap_between_imgs_end:all_underline_pos[3]]

    # number of labelled samples
    no_labelled_samples_start = dataset_tag.find('N')
    no_labelled_samples_end = no_labelled_samples_start + len('N')
    no_labelled_samples = dataset_tag[no_labelled_samples_end:all_underline_pos[4]]

    save_folder_dataset = data_directory + dataset_name + '/' + data_set_tag
    save_folder_resolution = save_folder_dataset + '/R' + resolution_imgs
    save_folder_case_no = save_folder_resolution + '/C' + case_no
    save_folder_patch_size = save_folder_case_no + '/D' + new_depth_of_img + '_S' + gap_between_imgs
    save_folder_patch_no = save_folder_patch_size + '/N' + no_labelled_samples
    save_folder_labelled = save_folder_patch_no + '/labelled'

    # unlabelled data
    # volume_string_start = dataset_tag.find('unlabelled')
    # volume_string_end = volume_string_start + len('unlabelled')
    # unlabelled_no = dataset_tag[volume_string_end:]

    train_image_folder_labelled = save_folder_labelled + '/patches'
    train_label_folder_labelled = save_folder_labelled + '/labels'
    train_dataset_labelled = CustomDataset(train_image_folder_labelled, train_label_folder_labelled, data_augmentation, labelled=True)

    # train_image_folder_unlabelled = save_folder_patch_size + '/unlabelled' + unlabelled_no + '/patches'
    # train_label_folder_unlabelled = save_folder_patch_size + '/unlabelled' + unlabelled_no + '/labels'
    # train_dataset_unlabelled = CustomDataset(train_image_folder_unlabelled, train_label_folder_unlabelled, 'none', labelled=True)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=False)
    # trainloader_unlabelled = data.DataLoader(train_dataset_unlabelled, batch_size=train_batchsize*ratio, shuffle=True, num_workers=0, drop_last=False)

    validate_image_folder = save_folder_patch_size + '/validate/patches'
    validate_label_folder = save_folder_patch_size + '/validate/labels'
    test_image_folder = save_folder_patch_size + '/test/patches'
    test_label_folder = save_folder_patch_size + '/test/labels'

    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, 'none', labelled=True)
    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none', labelled=True)

    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    return trainloader_labelled, validateloader, testloader, train_dataset_labelled, validate_dataset, test_dataset


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, data_augmentation='none', multi_class=False):

    if multi_class is True:
        multi_class_tag = '3d_multi'
    else:
        multi_class_tag = '3d_binary'

    data_directory = data_directory + dataset_name + '/' + dataset_tag + '/' + multi_class_tag
    data_directory_eval_test = data_directory + dataset_name

    folder_labelled = data_directory + '/train'

    train_image_folder_labelled = folder_labelled + '/patches'
    train_label_folder_labelled = folder_labelled + '/labels'
    train_dataset_labelled = CustomDataset(train_image_folder_labelled, train_label_folder_labelled, data_augmentation, labelled=True)

    # train_image_folder_unlabelled = data_directory + '/unlabelled/patches'
    # train_label_folder_unlabelled = data_directory + '/unlabelled/labels'
    # train_dataset_unlabelled = CustomDataset(train_image_folder_unlabelled, train_label_folder_unlabelled, 'none', labelled=True)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=True)
    # trainloader_unlabelled = data.DataLoader(train_dataset_unlabelled, batch_size=train_batchsize*ratio, shuffle=True, num_workers=0, drop_last=False)

    validate_image_folder = data_directory + '/validate/patches'
    validate_label_folder = data_directory + '/validate/labels'
    test_image_folder = data_directory + '/test/patches'
    test_label_folder = data_directory + '/test/labels'

    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, 'none', labelled=True)
    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none', labelled=True)

    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    return trainloader_labelled, validateloader, testloader, train_dataset_labelled, validate_dataset, test_dataset
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
                     testdata,
                     log_tag,
                     class_no,
                     lr_decay):

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + dataset_tag + '/' + log_tag
    os.mkdir(saved_information_path, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    os.mkdir(saved_log_path, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    os.mkdir(saved_model_path, exist_ok=True)

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-5)

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_decay == 'poly':
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)
        elif lr_decay == 'steps:':
            for param_group in optimizer.param_groups:
                if step % (num_steps // 4) == 0:
                    param_group["lr"] = param_group["lr"] * 0.5
        elif lr_decay == 'warmup':
            warm_up_lr = 0.8
            for param_group in optimizer.param_groups:
                if step < num_steps*warm_up_lr:
                    param_group["lr"] = step * learning_rate / (num_steps*warm_up_lr)
                else:
                    param_group["lr"] = learning_rate
        else:
            pass

        print(
            'Step [{}/{}], '
            'lr: {:.4f},'
            'Train sup loss: {:.4f}, '
            'Train iou: {:.4f}, '
            'val iou:{:.4f}, '.format(step + 1, num_steps,
                                      optimizer.param_groups[0]["lr"],
                                      np.nanmean(train_sup_loss),
                                      np.nanmean(train_iou),
                                      np.nanmean(validate_iou)))

        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #

        writer.add_scalars('acc metrics', {'train iou': np.nanmean(train_iou),
                                           'val hausdorff dist': np.nanmean(validate_h_dist),
                                           'val iou': np.nanmean(validate_iou)}, step + 1)

        writer.add_scalars('loss values', {'sup loss': np.nanmean(train_sup_loss)}, step + 1)

        # if step > num_steps - 10:
        #     save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
        #     path_model = save_model_name_full
        #     torch.save(model, path_model)

    save_model_name_full = saved_model_path + '/' + save_model_name + '.pt'
    path_model = save_model_name_full
    torch.save(model, path_model)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    test_iou, test_h_dist, test_recall, test_precision = test(saved_information_path, saved_model_path, testdata, device, model_name, class_no, training_time)

    print('Test IoU: ' + str(np.nanmean(test_iou)) + '\n')
    print('Test IoU std: ' + str(np.nanstd(test_iou)) + '\n')
    print('Test H-dist: ' + str(np.nanmean(test_h_dist)) + '\n')
    print('Test H-dist std: ' + str(np.nanstd(test_h_dist)) + '\n')
    print('Test recall: ' + str(np.nanmean(test_recall)) + '\n')
    print('Test recall std: ' + str(np.nanstd(test_recall)) + '\n')
    print('Test precision: ' + str(np.nanmean(test_precision)) + '\n')
    print('Test precision std: ' + str(np.nanstd(test_precision)) + '\n')

    print('\nTraining finished and model saved\n')
    # zip all models:
    saved_information_path_all_models = saved_information_path + '/all_models'
    shutil.make_archive(saved_information_path_all_models, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
