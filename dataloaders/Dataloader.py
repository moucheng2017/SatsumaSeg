import glob
import os
import random

import torch
import numpy as np

# Read cases, for each case, read the whole volume
# 1. random crop around the lung mask
# 2. crop again around the lung mask but with a lower resolution
# 3. apply contrast augmentation and noises to create another input
# Right now it is just random cropping


class RandomCropping(object):
    def __init__(self, output_size, skip_slices):
        # output_size: d x h x w
        self.output_size = output_size
        self.skip_slices = skip_slices

    def crop(self, *volumes):
        # for supervised learning, we need to crop both volume and the label, arg1: volume, arg2: label
        # for unsupervised learning, we only need to crop the volume, arg1: volume
        # new_resolution = 224
        new_d = self.output_size[0]
        new_h = self.output_size[1]
        new_w = self.output_size[2]

        skip = random.choice(self.skip_slices)
        skip_x = random.choice(self.skip_slices)

        for volume in volumes:
            c, d, h, w = np.shape(volume)
            assert d > new_d

        if h > new_h:
            top_h = np.random.randint(0, h - new_h)
            top_w = np.random.randint(0, w - new_w)

        top_d = np.random.randint(0, d - skip*new_d)
        outputs = []

        for each_input in volumes:
            # skip every 3 slices, so equivalently, we look at longer span:
            # each_input = each_input[:, ::ratio, :, :]
            if h > new_h:
                 each_output = each_input[
                               :,
                               top_d:top_d+skip*new_d:skip,
                               top_h:top_h+new_h,
                               top_w:top_w+new_w
                               ]
            else:
                 each_output = each_input[
                               :,
                               top_d:top_d+skip*new_d:skip,
                               :,
                               :
                               ]
            outputs.append(each_output)

        return outputs


class RandomContrast(object):
    def __init__(self, bin_range=[10, 256]):
        self.bin_low = bin_range[0]
        self.bin_high = bin_range[1]

    def randomintensity(self, input):
        bin = np.random.randint(self.bin_low, self.bin_high)
        c, d, h, w = np.shape(input)
        for each_slice in range(d):
            single_channel = input[:, each_slice, :, :].squeeze()
            image_histogram, bins = np.histogram(single_channel.flatten(), bin, density=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            single_channel = np.interp(single_channel.flatten(), bins[:-1], cdf)
            input[:, each_slice, :, :] = np.reshape(single_channel, (c, 1, h, w))
        return input


class RandomGaussian(object):
    def __init__(self, mean=0, std=0.1):
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


class CT_Dataset(torch.utils.data.Dataset):
    '''
    Each volume should be at: Dimension X Height X Width
    '''
    def __init__(self, imgs_folder, labels_folder, new_size, labelled):
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.labelled_flag = labelled
        self.augmentation_contrast = RandomContrast()
        self.augmentation_cropping = RandomCropping(new_size, [1, 2])
        self.augmentation_gaussian = RandomGaussian()

    def __getitem__(self, index):
        all_images = glob.glob(os.path.join(self.imgs_folder, '*.npy'))
        all_images.sort()
        imagename = all_images[index]
        image = np.load(imagename)
        image = np.array(image, dtype='float32')
        # print(np.shape(image))

        all_labels = glob.glob(os.path.join(self.labels_folder, '*.npy'))
        all_labels.sort()
        label = np.load(all_labels[index])
        label = np.array(label, dtype='float32')
        # print(np.shape(label))

        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        if self.labelled_flag is True:
            [image, label] = self.augmentation_cropping.crop(image, label)
            image1 = self.augmentation_contrast.randomintensity(image)
            # image2 = self.augmentation_gaussian.gaussiannoise(image)
            weights = np.random.dirichlet((1, 1), 1)
            image = weights[0][0]*image + weights[0][1]*image1
            return image, label, imagename
        else:
            [image] = self.augmentation_cropping.crop(image)
            image1 = self.augmentation_contrast.randomintensity(image)
            # image2 = self.augmentation_gaussian.gaussiannoise(image)
            weights = np.random.dirichlet((1, 1), 1)
            image = weights[0][0]*image + weights[0][1]*image1
            return image, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))


if __name__ == '__main__':
    dummy_input = np.random.rand(512, 512, 480)
    cropping_augmentation = RandomCropping(64, [1])
    output = cropping_augmentation.crop(dummy_input, dummy_input)
    print(np.shape(output))


