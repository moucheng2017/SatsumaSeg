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
from dataloaders.Dataloader import CT_Dataset
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
                new_resolution=[12, 224, 224],
                l2=0.01
                ):

    for j in range(1, repeat + 1):

        repeat_str = str(j)

        Exp = Unet3D(in_ch=input_dim, width=width, class_no=class_no, z_downsample=downsample)
        Exp_name = 'sup_unet' + \
                   '_e' + str(repeat_str) + \
                   '_l' + str(learning_rate) + \
                   '_b' + str(train_batchsize) + \
                   '_w' + str(width) + \
                   '_s' + str(num_steps) + \
                   '_d' + str(downsample) + \
                   '_r' + str(l2) + \
                   '_z' + str(new_resolution[0]) + \
                   '_x' + str(new_resolution[1])

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
                         l2=l2,
                         dilation=1)


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, new_resolution):

    data_directory = data_directory + dataset_name + '/' + dataset_tag
    data_directory_eval_test = data_directory + dataset_name

    folder_labelled = data_directory + '/labelled'

    train_image_folder_labelled = folder_labelled + '/imgs'
    train_label_folder_labelled = folder_labelled + '/lbls'
    train_dataset_labelled = CT_Dataset(train_image_folder_labelled, train_label_folder_labelled, new_resolution, labelled=True)

    # train_image_folder_unlabelled = data_directory + '/unlabelled/patches'
    # train_label_folder_unlabelled = data_directory + '/unlabelled/labels'
    # train_dataset_unlabelled = CustomDataset(train_image_folder_unlabelled, train_label_folder_unlabelled, 'none', labelled=True)

    trainloader_labelled = data.DataLoader(train_dataset_labelled, batch_size=train_batchsize, shuffle=True, num_workers=0, drop_last=True)
    # trainloader_unlabelled = data.DataLoader(train_dataset_unlabelled, batch_size=train_batchsize*ratio, shuffle=True, num_workers=0, drop_last=False)

    validate_image_folder = data_directory + '/validate/imgs'
    validate_label_folder = data_directory + '/validate/lbls'

    testdata_path = data_directory + '/test'

    test_image_folder = data_directory + '/test/imgs'
    test_label_folder = data_directory + '/test/lbls'

    validate_dataset = CT_Dataset(validate_image_folder, validate_label_folder, new_resolution, labelled=True)
    test_dataset = CT_Dataset(test_image_folder, test_label_folder, new_resolution, labelled=True)

    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
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
                     l2,
                     class_no):

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + dataset_tag + '/' + log_tag
    if not os.path.exists(saved_information_path):
        os.makedirs(saved_information_path, exist_ok=True)
    # os.mkdir(saved_information_path, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    # os.mkdir(saved_log_path, exist_ok=True)
    if not os.path.exists(saved_log_path):
        os.makedirs(saved_log_path, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    # os.mkdir(saved_model_path, exist_ok=True)
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

        outputs, _ = model(train_imgs, [dilation, dilation, dilation, dilation], [dilation, dilation, dilation, dilation])
        if class_no == 2:
            prob_outputs = torch.sigmoid(outputs)
        else:
            prob_outputs = F.softmax(outputs, dim=1)

        if class_no == 2:
            loss = SoftDiceLoss()(prob_outputs, labels) + nn.BCELoss(reduction='mean')(prob_outputs.squeeze(), labels.squeeze())
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

        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)

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
                                           # 'val hausdorff dist': np.nanmean(validate_h_dist),
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

    # test_image_path = os.path.join(testdata_path, 'patches')
    # test_label_path = os.path.join(testdata_path, 'labels')
    # test_iou = test(saved_information_path + '/' + save_model_name,
    #                 saved_model_path,
    #                 test_image_path,
    #                 test_label_path,
    #                 device,
    #                 model_name,
    #                 class_no,
    #                 [192, 192, 192],
    #                 1)
    #
    # print('Test IoU: ' + str(np.nanmean(test_iou)) + '\n')
    # print('Test IoU std: ' + str(np.nanstd(test_iou)) + '\n')

    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
