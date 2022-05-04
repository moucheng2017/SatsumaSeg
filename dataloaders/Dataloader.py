import glob
import os
import random

import torch
import numpy as np

import nibabel as nib

import numpy.ma as ma


class RandomCropping(object):
    def __init__(self, output_size,
                       skip_slices):

        # output_size: d x h x w
        self.output_size = output_size
        self.skip_slices = skip_slices

    def crop(self, *volumes):
        # for supervised learning, we need to crop both volume and the label, arg1: volume, arg2: label
        # for unsupervised learning, we only need to crop the volume, arg1: volume
        # new_resolution = 224
        # begining slices and ending slices are not informative so discarded:

        discared_slices = 10

        for volume in volumes:
            c, d, h, w = np.shape(volume)

        if self.output_size is None:
            new_d = 1
            new_h = h
            new_w = w
        else:
            new_d = self.output_size[0]
            new_h = self.output_size[1]
            new_w = self.output_size[2]

        skip = random.choice(self.skip_slices)
        # skip_x = random.choice(self.skip_slices)

        if h > new_h:
            top_h = np.random.randint(0, h - new_h)
            top_w = np.random.randint(0, w - new_w)

        top_d = np.random.randint(skip*new_d+discared_slices, d - skip*new_d-discared_slices)
        outputs = []

        for each_input in volumes:
            # skip every 3 slices, so equivalently, we look at longer span:
            # each_input = each_input[:, ::ratio, :, :]
            if h > new_h:
                if new_d > 1:
                     each_output = each_input[
                                   :,
                                   top_d:top_d+skip*new_d:skip,
                                   top_h:top_h+new_h,
                                   top_w:top_w+new_w
                                   ]
                else:
                    each_output = each_input[
                                  :,
                                  top_d,
                                  top_h:top_h + new_h,
                                  top_w:top_w + new_w
                                  ]

            else:
                if new_d > 1:
                     each_output = each_input[
                                   :,
                                   top_d:top_d+skip*new_d:skip,
                                   :,
                                   :
                                   ]
                else:
                    each_output = each_input[
                                  :,
                                  top_d,
                                  :,
                                  :
                                  ]
                    # print(np.shape(each_output))
            outputs.append(each_output)

        return outputs


class RandomContrast(object):
    def __init__(self, bin_range=[100, 255]):
        # self.bin_low = bin_range[0]
        # self.bin_high = bin_range[1]
        self.bin_range = bin_range

    def randomintensity(self, input):

        augmentation_flag = np.random.rand()

        if augmentation_flag >= 0.5:
            # bin = np.random.choice(self.bin_range)
            bin = random.randint(self.bin_range[0], self.bin_range[1])
            c, d, h, w = np.shape(input)
            image_histogram, bins = np.histogram(input.flatten(), bin, density=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            output = np.interp(input.flatten(), bins[:-1], cdf)
            output = np.reshape(output, (c, d, h, w))
        else:
            output = input

        return output


class RandomGaussian(object):
    def __init__(self, mean=0, std=0.01):
        self.m = mean
        self.sigma = std

    def gaussiannoise(self, input):
        noise = np.random.normal(self.m, self.sigma, input.shape)
        mask_overflow_upper = input + noise >= 1.0
        mask_overflow_lower = input + noise < 0.0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0.0
        input += noise
        return input


def normalisation(lung, image):
    image_masked = ma.masked_where(lung > 0.5, image)
    lung_mean = np.nanmean(image_masked)
    lung_std = np.nanstd(image_masked)
    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


class CT_Dataset(torch.utils.data.Dataset):
    '''
    Each volume should be at: Dimension X Height X Width
    '''
    def __init__(self, imgs_folder, labels_folder, lung_folder, new_size, labelled):
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.lung_folder = lung_folder
        # self.dimension = dim

        self.labelled_flag = labelled
        self.augmentation_contrast = RandomContrast()
        self.augmentation_cropping = RandomCropping(new_size, [1])
        # self.augmentation_gaussian = RandomGaussian()

    def __getitem__(self, index):
        # Lung masks:
        all_lungs = sorted(glob.glob(os.path.join(self.lung_folder, '*.nii.gz*')))
        lung = nib.load(all_lungs[index])
        lung = lung.get_fdata()
        lung = np.array(lung, dtype='float32')
        lung = np.transpose(lung, (2, 0, 1))
        lung = np.expand_dims(lung, axis=0)

        # Images:
        all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
        imagename = all_images[index]

        # load image and preprocessing:
        image = nib.load(imagename)
        image = image.get_fdata()
        image = np.array(image, dtype='float32')

        # transform dimension:
        # original dimension: (H x W x D)
        image = np.transpose(image, (2, 0, 1))
        # (D x H x W)
        image = np.expand_dims(image, axis=0)
        # (C x D x H x W)

        # Now applying lung window:
        image[image < -1000.0] = -1000.0
        image[image > 500.0] = 500.0

        # Apply normalisation with values inside of lung
        image = normalisation(lung, image)

        # Random contrast and Renormalisation:
        # image_diff_contrast = self.augmentation_contrast.randomintensity(image)
        # image = 0.7*image + 0.3*image_diff_contrast
        # image = normalisation(lung, image)

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        if self.labelled_flag is True:
            # Labels:
            all_labels = sorted(glob.glob(os.path.join(self.labels_folder, '*.nii.gz*')))
            label = nib.load(all_labels[index])
            label = label.get_fdata()
            label = np.array(label, dtype='float32')
            label = np.transpose(label, (2, 0, 1))
            label = np.expand_dims(label, axis=0)
            [image, label, lung] = self.augmentation_cropping.crop(image, label, lung)
            return image, label, lung, imagename
        else:
            [image, lung] = self.augmentation_cropping.crop(image, lung)
            return image, lung, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))


if __name__ == '__main__':
    dummy_input = np.random.rand(512, 512, 480)
    cropping_augmentation = RandomCropping(64, [1])
    output = cropping_augmentation.crop(dummy_input, dummy_input)
    print(np.shape(output))


