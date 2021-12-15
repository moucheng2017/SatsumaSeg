import glob
import os
import torch
# import gzip
# import shutil
# import random
import errno
# # import pydicom
import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# from nipype.interfaces.ants import N4BiasFieldCorrection
from tifffile import imsave


# todo:
# Read cases, for each case, read the whole volume
# 1. random crop around the lung mask
# 2. crop again around the lung mask but with a lower resolution
# 3. apply contrast augmentation and noises to create another input

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
