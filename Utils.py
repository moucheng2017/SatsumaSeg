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
import torch.nn.functional as F

import glob
# import tifffile as tiff

from scipy import ndimage
import random

from skimage import exposure

from Metrics import segmentation_scores, hd95, preprocessing_accuracy, f1_score
from PIL import Image
from torch.utils import data


def sigmoid_rampup(current, rampup_length, limit):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    phase = 1.0 - current / rampup_length
    weight = float(np.exp(-5.0 * phase * phase))
    if weight > limit:
        return float(limit)
    else:
        return weight


def cyclic_sigmoid_rampup(current, rampup_length, limit):
    # calculate the relative current:
    cyclic_index = current // rampup_length
    relative_current = current - cyclic_index*rampup_length
    phase = 1.0 - relative_current / rampup_length
    weight = float(np.exp(-5.0 * phase * phase))
    if weight > limit:
        return float(limit)
    else:
        return weight


def exp_rampup(current, base, limit):
    weight = float(base*(1.05**current))
    if weight > limit:
        return float(limit)
    else:
        weight


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_folder, labels_folder, augmentation, labelled=True, volume=True):

        # 1. Initialize file paths or a list of file names.
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.data_augmentation = augmentation
        self.dimension = volume
        self.labelled_flag = labelled

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using num py.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        all_images = glob.glob(os.path.join(self.imgs_folder, '*.npy'))
        all_images.sort()
        image = np.load(all_images[index])
        image = np.array(image, dtype='float32')

        c_amount = len(np.shape(image))
        # the image is expecrted to be in format: channel x dimension x height x width
        if c_amount == 3:
            d1, d2, d3 = np.shape(image)
            if d2 != d3:
                # dimension should be: channel x height x width
                image = np.transpose(image, (2, 0, 1))
            if self.dimension is True:
                # sometimes, when the channel is one, image is saved as dimension x height x width so we need to expand one channel dimension
                # check dimension
                image = np.expand_dims(image, axis=0)
        elif c_amount == 4:
            if self.dimension is True:
                pass
            else:
                image = np.squeeze(image, axis=0)
        else:
            pass

        # print(np.shape(image))

        imagename = all_images[index]
        path_label, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        if self.labelled_flag is True:
            all_labels = glob.glob(os.path.join(self.labels_folder, '*.npy'))
            all_labels.sort()
            label = np.load(all_labels[index])
            label_origin = np.array(label, dtype='float32')
            # Reshaping everyting to make sure the order: channel x height x width
            c_amount = len(np.shape(label))
            if c_amount == 3:
                d1, d2, d3 = np.shape(label)
                if d2 != d3:
                    label_origin = np.transpose(label_origin, (2, 0, 1))
                if self.dimension is True:
                    label = np.expand_dims(label_origin, axis=0)
            elif c_amount == 4:
                if self.dimension is True:
                    pass
                else:
                    label = np.squeeze(label, axis=0)
            else:
                pass
        else:
            pass

        if self.data_augmentation == 'gaussian':
            # gaussian noises:
            augmentation = random.uniform(0, 1)
            if augmentation >= 0.5:
                mean = 0.0
                sigma = 0.1
                noise = np.random.normal(mean, sigma, image.shape)
                mask_overflow_upper = image + noise >= 1.0
                mask_overflow_lower = image + noise < 0.0
                noise[mask_overflow_upper] = 1.0
                noise[mask_overflow_lower] = 0.0
                image += noise
            else:
                pass

        elif self.data_augmentation == 'contrast':
            # random contrast
            augmentation = random.uniform(0, 1)
            bin = random.choice([10, 30, 50, 75, 100, 125, 150, 175, 200, 250])
            if self.dimension is True:
                c, d, h, w = np.shape(image)
                for each_slice in range(d):
                    single_channel = image[:, each_slice, :, :].squeeze()
                    image_histogram, bins = np.histogram(single_channel.flatten(), bin, density=True)
                    cdf = image_histogram.cumsum()  # cumulative distribution function
                    cdf = 255 * cdf / cdf[-1]  # normalize
                    single_channel = np.interp(single_channel.flatten(), bins[:-1], cdf)
                    image[:, each_slice, :, :] = np.reshape(single_channel, (c, 1, h, w))
            else:
                c, h, w = np.shape(image)
                for each_slice in range(c):
                    single_channel = image[each_slice, :, :].squeeze()
                    image_histogram, bins = np.histogram(single_channel.flatten(), bin, density=True)
                    cdf = image_histogram.cumsum()  # cumulative distribution function
                    cdf = 255 * cdf / cdf[-1]  # normalize
                    single_channel = np.interp(single_channel.flatten(), bins[:-1], cdf)
                    image[each_slice, :, :] = np.reshape(single_channel, (1, h, w))

        elif self.data_augmentation == 'all':
            # random contrast + Gaussian
            bin = random.choice([10, 30, 50, 75, 100, 125, 150, 175, 200, 250])

            if self.dimension is True:
                c, d, h, w = np.shape(image)
                for each_slice in range(d):
                    single_channel = image[:, each_slice, :, :].squeeze()
                    image_histogram, bins = np.histogram(single_channel.flatten(), bin, density=True)
                    cdf = image_histogram.cumsum()  # cumulative distribution function
                    cdf = 255 * cdf / cdf[-1]  # normalize
                    single_channel = np.interp(single_channel.flatten(), bins[:-1], cdf)
                    image[:, each_slice, :, :] = np.reshape(single_channel, (c, 1, h, w))
            else:
                c, h, w = np.shape(image)
                for each_slice in range(c):
                    single_channel = image[each_slice, :, :].squeeze()
                    image_histogram, bins = np.histogram(single_channel.flatten(), bin, density=True)
                    cdf = image_histogram.cumsum()  # cumulative distribution function
                    cdf = 255 * cdf / cdf[-1]  # normalize
                    single_channel = np.interp(single_channel.flatten(), bins[:-1], cdf)
                    image[each_slice, :, :] = np.reshape(single_channel, (1, h, w))

            mean = 0.0
            sigma = 0.1
            noise = np.random.normal(mean, sigma, image.shape)
            mask_overflow_upper = image + noise >= 1.0
            mask_overflow_lower = image + noise < 0.0
            noise[mask_overflow_upper] = 1.0
            noise[mask_overflow_lower] = 0.0
            image += noise

        else:
            pass

        if self.labelled_flag is True:
            return image, label, imagename
        else:
            return image, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))


def evaluate(validateloader, model, device, model_name, class_no, dilation):

    model.eval()
    with torch.no_grad():

        validate_iou = []
        validate_h_dist = []

        for i, (val_images, val_label, imagename) in enumerate(validateloader):
            val_img = val_images.to(device=device, dtype=torch.float32)
            # val_img = val_images.to(device=device, dtype=torch.float32).unsqueeze(1)
            val_label = val_label.to(device=device, dtype=torch.float32)

            if 'CCT' in model_name or 'cct' in model_name:
                val_outputs, _ = model(val_img)
            elif 'expert' in model_name:
                val_outputs = model(val_img, dilation)
            else:
                val_outputs = model(val_img)

            if class_no == 2:
                val_outputs = torch.sigmoid(val_outputs)
                val_class_outputs = (val_outputs > 0.5).float()
            else:
                _, val_class_outputs = torch.max(val_outputs, dim=1)

            eval_mean_iu_ = segmentation_scores(val_label.squeeze(), val_class_outputs.squeeze(), class_no)
            validate_iou.append(eval_mean_iu_)
            if (val_class_outputs == 1).sum() > 1 and (val_label == 1).sum() > 1:
                v_dist_ = hd95(val_class_outputs.squeeze(), val_label.squeeze(), class_no)
                validate_h_dist.append(v_dist_)

    return validate_iou, validate_h_dist


def test(saved_information_path, saved_model_path, testdata, device, model_name, class_no, training_time, dilation=16):

    save_path = saved_information_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    all_models = [os.path.join(saved_model_path, f) for f in listdir(saved_model_path) if os.path.isfile(os.path.join(saved_model_path, f))]
    all_models.sort()

    # all_models = glob.glob(os.path.join(saved_model_path, '*.pt'))
    # print(all_models)

    # testing acc with main decoder:
    test_iou = []
    # test_iou_wt = []
    # test_iou_et = []
    # test_iou_tc = []

    test_h_dist = []
    test_recall = []
    test_precision = []

    for model in all_models:
        # test_time = 0
        model = torch.load(model)
        model.eval()

        start = timeit.default_timer()
        with torch.no_grad():

            for ii, (test_images, test_label, test_imagename) in enumerate(testdata):
                test_img = test_images.to(device=device, dtype=torch.float32)
                # test_img = test_images.to(device=device, dtype=torch.float32).unsqueeze(1)
                test_label = test_label.to(device=device, dtype=torch.float32)

                assert torch.max(test_label) != 100.0

                if 'CCT' in model_name or 'cct' in model_name:
                    test_outputs, _ = model(test_img)
                elif 'expert' in model_name:
                    test_outputs = model(test_img, dilation)
                else:
                    test_outputs = model(test_img)

                if class_no == 2:
                    test_class_outputs = torch.sigmoid(test_outputs)
                    test_class_outputs = (test_class_outputs > 0.5).float()
                else:
                    _, test_class_outputs = torch.max(test_outputs, dim=1)

                # testing on average metrics:
                test_label = test_label.squeeze()
                test_class_outputs = test_class_outputs.squeeze()

                # # whole tumour: 1 == 1, 2 == 1, 3 == 1
                # test_label_wt = torch.zeros_like(test_label)
                # test_class_outputs_wt = torch.zeros_like(test_class_outputs)
                # test_label_wt[test_label == 1] = 1
                # test_label_wt[test_label == 2] = 1
                # test_label_wt[test_label == 3] = 1
                # test_class_outputs_wt[test_class_outputs == 1] = 1
                # test_class_outputs_wt[test_class_outputs == 2] = 1
                # test_class_outputs_wt[test_class_outputs == 3] = 1
                # test_mean_iu_wt_ = segmentation_scores(test_label_wt.squeeze(), test_class_outputs_wt.squeeze(), 2)
                # test_iou_wt.append(test_mean_iu_wt_)
                #
                # # enhancing tumour core: 3 == 1, 1 == 0, 2 == 0
                # test_label_et = torch.zeros_like(test_label)
                # test_class_outputs_et = torch.zeros_like(test_class_outputs)
                # test_label_et[test_label == 3] = 1
                # test_label_et[test_label == 1] = 0
                # test_label_et[test_label == 2] = 0
                # test_class_outputs_et[test_class_outputs == 3] = 1
                # test_class_outputs_et[test_class_outputs == 1] = 0
                # test_class_outputs_et[test_class_outputs == 2] = 0
                # test_mean_iu_et_ = segmentation_scores(test_label_wt.squeeze(), test_class_outputs_wt.squeeze(), 2)
                # test_iou_et.append(test_mean_iu_et_)
                #
                # # tumour core: 3 == 1, 1 == 1, 2 == 0
                # test_label_tc = torch.zeros_like(test_label)
                # test_class_outputs_tc = torch.zeros_like(test_class_outputs)
                # test_label_tc[test_label == 3] = 1
                # test_label_tc[test_label == 1] = 1
                # test_label_tc[test_label == 2] = 0
                # test_class_outputs_tc[test_class_outputs == 3] = 1
                # test_class_outputs_tc[test_class_outputs == 1] = 1
                # test_class_outputs_tc[test_class_outputs == 2] = 0
                # test_mean_iu_tc_ = segmentation_scores(test_label_tc.squeeze(), test_class_outputs_tc.squeeze(), 2)
                # test_iou_tc.append(test_mean_iu_tc_)

                # test_mean_iu_ = (test_mean_iu_tc_ + test_mean_iu_et_ + test_mean_iu_wt_) / 3
                test_mean_iu_ = segmentation_scores(test_label.squeeze(), test_class_outputs.squeeze(), 2)
                test_mean_f1_, test_mean_recall_, test_mean_precision_ = f1_score(test_label.squeeze(), test_class_outputs.squeeze(), class_no)

                test_iou.append(test_mean_iu_)
                test_recall.append(test_mean_recall_)
                test_precision.append(test_mean_precision_)

                # if (test_class_outputs == 1).sum() > 1 and (test_label == 1).sum() > 1:
                #     t_dist_ = hd95(test_class_outputs.squeeze(), test_label.squeeze(), class_no)
                #     test_h_dist.append(t_dist_)

        stop = timeit.default_timer()
        test_time = stop - start

    # print(len(testdata))
    result_dictionary = {
        'Test IoU mean': str(np.nanmean(test_iou)),
        'Test IoU std': str(np.nanstd(test_iou)),
        # 'Test IoU wt mean': str(np.nanmean(test_iou_wt)),
        # 'Test IoU wt std': str(np.nanstd(test_iou_wt)),
        # 'Test IoU et mean': str(np.nanmean(test_iou_et)),
        # 'Test IoU et std': str(np.nanstd(test_iou_et)),
        # 'Test IoU tc mean': str(np.nanmean(test_iou_tc)),
        # 'Test IoU tc std': str(np.nanstd(test_iou_tc)),
        'Test H-dist mean': str(np.nanmean(test_h_dist)),
        'Test H-dist std': str(np.nanstd(test_h_dist)),
        'Test recall mean': str(np.nanmean(test_recall)),
        'Test recall std': str(np.nanstd(test_recall)),
        'Test precision mean': str(np.nanmean(test_precision)),
        'Test precision std': str(np.nanstd(test_precision)),
        # 'Training time(s)': str(training_time),
        'Test time(s)': str(test_time / len(testdata))
    }

    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    iou_path = save_path + '/iou.csv'
    # iou_tc_path = save_path + '/iou_tc.csv'
    # iou_wt_path = save_path + '/iou_wt.csv'
    # iou_et_path = save_path + '/iou_et.csv'

    # h_dist_path = save_path + '/h_dist.csv'
    recall_path = save_path + '/recall.csv'
    precision_path = save_path + '/precision.csv'

    np.savetxt(iou_path, test_iou, delimiter=',')
    # np.savetxt(iou_wt_path, test_iou_wt, delimiter=',')
    # np.savetxt(iou_tc_path, test_iou_tc, delimiter=',')
    # np.savetxt(iou_et_path, test_iou_et, delimiter=',')
    # np.savetxt(h_dist_path, test_h_dist, delimiter=',')
    np.savetxt(recall_path, test_recall, delimiter=',')
    np.savetxt(precision_path, test_precision, delimiter=',')

    return test_iou, test_h_dist, test_recall, test_precision


def test_brats(saved_information_path, saved_model_path, testdata, device, model_name, class_no, training_time, dilation=16):

    save_path = saved_information_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    all_models = [os.path.join(saved_model_path, f) for f in listdir(saved_model_path) if os.path.isfile(os.path.join(saved_model_path, f))]
    all_models.sort()

    # all_models = glob.glob(os.path.join(saved_model_path, '*.pt'))
    # print(all_models)

    # testing acc with main decoder:
    test_iou = []
    test_iou_wt = []
    test_iou_et = []
    test_iou_tc = []

    test_h_dist = []
    test_recall = []
    test_precision = []

    for model in all_models:
        # test_time = 0
        model = torch.load(model)
        model.eval()

        start = timeit.default_timer()
        with torch.no_grad():

            for ii, (test_images, test_label, test_imagename) in enumerate(testdata):
                test_img = test_images.to(device=device, dtype=torch.float32)
                # test_img = test_images.to(device=device, dtype=torch.float32).unsqueeze(1)
                test_label = test_label.to(device=device, dtype=torch.float32)

                assert torch.max(test_label) != 100.0

                if 'CCT' in model_name or 'cct' in model_name:
                    test_outputs, _ = model(test_img)
                elif 'expert' in model_name:
                    test_outputs = model(test_img, dilation)
                else:
                    test_outputs = model(test_img)

                if class_no == 2:
                    test_class_outputs = torch.sigmoid(test_outputs)
                    test_class_outputs = (test_class_outputs > 0.5).float()
                else:
                    # _, test_class_outputs = torch.max(test_outputs, dim=1)
                    test_class_outputs = torch.sigmoid(test_outputs)
                    test_class_outputs = (test_class_outputs > 0.5).float()

                # testing on average metrics:
                # test_label = test_label.squeeze()
                # test_class_outputs = test_class_outputs.squeeze()

                # whole tumour: 1 == 1, 2 == 1, 3 == 1
                test_label_wt = torch.zeros_like(test_label)
                # test_class_outputs_wt = torch.zeros_like(test_class_outputs)
                test_label_wt[test_label == 1] = 1
                test_label_wt[test_label == 2] = 1
                test_label_wt[test_label == 3] = 1
                if len(test_class_outputs.size()) == 4:
                    test_class_outputs_wt = test_class_outputs[:, 0, :, :]
                else:
                    test_class_outputs_wt = test_class_outputs[:, :, 0, :, :]
                # test_class_outputs_wt[test_class_outputs == 1] = 1
                # test_class_outputs_wt[test_class_outputs == 2] = 1
                # test_class_outputs_wt[test_class_outputs == 3] = 1
                test_mean_iu_wt_ = segmentation_scores(test_label_wt.squeeze(), test_class_outputs_wt.squeeze(), 2)
                test_iou_wt.append(test_mean_iu_wt_)

                # tumour core: 3 == 1, 1 == 1, 2 == 0
                test_label_tc = torch.zeros_like(test_label)
                # test_class_outputs_tc = torch.zeros_like(test_class_outputs)
                test_label_tc[test_label == 3] = 1
                test_label_tc[test_label == 1] = 1
                test_label_tc[test_label == 2] = 0
                if len(test_class_outputs.size()) == 4:
                    test_class_outputs_tc = test_class_outputs[:, 1, :, :]
                else:
                    test_class_outputs_tc = test_class_outputs[:, :, 1, :, :]
                # test_class_outputs_tc[test_class_outputs == 3] = 1
                # test_class_outputs_tc[test_class_outputs == 1] = 1
                # test_class_outputs_tc[test_class_outputs == 2] = 0
                test_mean_iu_tc_ = segmentation_scores(test_label_tc.squeeze(), test_class_outputs_tc.squeeze(), 2)
                test_iou_tc.append(test_mean_iu_tc_)

                # enhancing tumour core: 3 == 1, 1 == 0, 2 == 0
                test_label_et = torch.zeros_like(test_label)
                # test_class_outputs_et = torch.zeros_like(test_class_outputs)
                test_label_et[test_label == 3] = 1
                test_label_et[test_label == 1] = 0
                test_label_et[test_label == 2] = 0
                if len(test_class_outputs.size()) == 4:
                    test_class_outputs_et = test_class_outputs[:, 2, :, :]
                else:
                    test_class_outputs_et = test_class_outputs[:, :, 2, :, :]

                # test_class_outputs_et[test_class_outputs == 3] = 1
                # test_class_outputs_et[test_class_outputs == 1] = 0
                # test_class_outputs_et[test_class_outputs == 2] = 0
                test_mean_iu_et_ = segmentation_scores(test_label_et.squeeze(), test_class_outputs_et.squeeze(), 2)
                test_iou_et.append(test_mean_iu_et_)

                test_mean_iu_ = (test_mean_iu_tc_ + test_mean_iu_et_ + test_mean_iu_wt_) / 3
                # test_mean_iu_ = segmentation_scores(test_label.squeeze(), test_class_outputs.squeeze(), 2)
                # test_mean_f1_, test_mean_recall_, test_mean_precision_ = f1_score(test_label.squeeze(), test_class_outputs.squeeze(), class_no)

                test_iou.append(test_mean_iu_)
                # test_recall.append(test_mean_recall_)
                # test_precision.append(test_mean_precision_)

                # if (test_class_outputs == 1).sum() > 1 and (test_label == 1).sum() > 1:
                #     t_dist_ = hd95(test_class_outputs.squeeze(), test_label.squeeze(), class_no)
                #     test_h_dist.append(t_dist_)

        stop = timeit.default_timer()
        test_time = stop - start

    # print(len(testdata))
    result_dictionary = {
        'Test IoU mean': str(np.nanmean(test_iou)),
        'Test IoU std': str(np.nanstd(test_iou)),
        'Test IoU wt mean': str(np.nanmean(test_iou_wt)),
        'Test IoU wt std': str(np.nanstd(test_iou_wt)),
        'Test IoU et mean': str(np.nanmean(test_iou_et)),
        'Test IoU et std': str(np.nanstd(test_iou_et)),
        'Test IoU tc mean': str(np.nanmean(test_iou_tc)),
        'Test IoU tc std': str(np.nanstd(test_iou_tc)),
        # 'Test H-dist mean': str(np.nanmean(test_h_dist)),
        # 'Test H-dist std': str(np.nanstd(test_h_dist)),
        # 'Test recall mean': str(np.nanmean(test_recall)),
        # 'Test recall std': str(np.nanstd(test_recall)),
        # 'Test precision mean': str(np.nanmean(test_precision)),
        # 'Test precision std': str(np.nanstd(test_precision)),
        'Training time(s)': str(training_time),
        'Test time(s)': str(test_time / len(testdata))
    }

    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    iou_path = save_path + '/iou.csv'
    iou_tc_path = save_path + '/iou_tc.csv'
    iou_wt_path = save_path + '/iou_wt.csv'
    iou_et_path = save_path + '/iou_et.csv'

    # h_dist_path = save_path + '/h_dist.csv'
    # recall_path = save_path + '/recall.csv'
    # precision_path = save_path + '/precision.csv'

    np.savetxt(iou_path, test_iou, delimiter=',')
    # np.savetxt(iou_wt_path, test_iou_wt, delimiter=',')
    # np.savetxt(iou_tc_path, test_iou_tc, delimiter=',')
    # np.savetxt(iou_et_path, test_iou_et, delimiter=',')
    # np.savetxt(h_dist_path, test_h_dist, delimiter=',')
    # np.savetxt(recall_path, test_recall, delimiter=',')
    # np.savetxt(precision_path, test_precision, delimiter=',')

    return test_iou





