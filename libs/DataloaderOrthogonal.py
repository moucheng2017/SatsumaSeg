import glob
import os
import torch
import random
import numpy as np
import nibabel as nib
import numpy.ma as ma
from torch.utils import data
from torch.utils.data import Dataset


class RandomCroppingOrthogonal(object):
    def __init__(self,
                 discarded_slices=5,
                 resolution=512):
        '''
        cropping_d: 3 d dimension of cropped sub volume cropping on h x w
        cropping_h: 3 d dimension of cropped sub volume cropping on w x d
        cropping_w: 3 d dimension of cropped sub volume cropping on h x d
        '''
        self.discarded_slices = discarded_slices
        self.resolution = resolution

    def crop(self, *volumes):

        # for supervised learning, we need to crop both volume and the label, arg1: volume, arg2: label
        # for unsupervised learning, we only need to crop the volume, arg1: volume

        sample_position_d_d = np.random.randint(self.discarded_slices, self.resolution-1)
        sample_position_h_h = np.random.randint(self.discarded_slices, self.resolution-1)
        sample_position_w_w = np.random.randint(self.discarded_slices, self.resolution-1)

        outputs = {"plane_d": [],
                   "plane_h": [],
                   "plane_w": []}

        for each_input in volumes:

            newd, newh, neww = np.shape(each_input)

            assert newd == self.resolution
            assert newh == self.resolution
            assert neww == self.resolution

            outputs["plane_d"].append(each_input[sample_position_d_d, :, :])
            outputs["plane_h"].append(each_input[:, sample_position_h_h, :])
            outputs["plane_w"].append(each_input[:, :, sample_position_w_w])

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
            # c, d, h, w = np.shape(input)
            c, h, w = np.shape(input)
            image_histogram, bins = np.histogram(input.flatten(), bin, density=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            output = np.interp(input.flatten(), bins[:-1], cdf)
            output = np.reshape(output, (c, h, w))
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


class RandomCutOut(object):
    # In house implementation of cutout for segmentation.
    # We create a zero mask to cover up the same part of the image and the mask. Both BCE and Dice will have zero gradients
    # if both seg and mask value are zero at the same position.
    # This is only applied on segmentation loss!!
    def __int__(self, tensor, mask_height=50, mask_width=50):
        self.segmentation_tensor = tensor
        self.w_mask = mask_width
        self.h_mask = mask_height

    def cutout_seg(self, x, y):
        '''
        Args:
            x: segmentation
            y: gt
        Returns:
        '''
        b, c, h, w = self.segmentation_tensor.size()
        assert self.w_mask <= w
        assert self.h_mask <= h

        h_starting = np.random.randint(0, h - self.h_mask)
        w_starting = np.random.randint(0, w - self.h_mask)
        h_ending = h_starting + self.h_mask
        w_ending = w_starting + self.w_mask

        mask = torch.ones_like(self.segmentation_tensor).cuda()
        mask[:, :, h_starting:h_ending, w_starting:w_ending] = 0

        return x*mask, y*mask


class RandomCutMix(object):
    # In house implementation of cutmix for segmentation. This is inspired by the original cutmix but very different from the original one!!
    # We mix a part of image 1 with another part of image 2 and do the same for the paired labels.
    # This is applied before feeding into the network!!!
    def __int__(self, tensor, mask_height=50, mask_width=50):

        self.segmentation_tensor = tensor
        self.w_mask = mask_width
        self.h_mask = mask_height

    def cutmix_seg(self, x, y):
        '''
        Args:
            x: segmentation
            y: gt
        Returns:
        '''
        b, c, h, w = self.segmentation_tensor.size()

        assert self.w_mask <= w
        assert self.h_mask <= h

        h_starting = np.random.randint(0, h - self.h_mask)
        w_starting = np.random.randint(0, w - self.h_mask)
        h_ending = h_starting + self.h_mask
        w_ending = w_starting + self.w_mask

        index = np.random.permutation(b)
        x_2 = x[index, :]
        y_2 = y[index, :]

        x[:, :, h_starting:h_ending, w_starting:w_ending] = x_2[:, :, h_starting:h_ending, w_starting:w_ending]
        y[:, :, h_starting:h_ending, w_starting:w_ending] = y_2[:, :, h_starting:h_ending, w_starting:w_ending]

        return x, y


def normalisation(lung, image, apply_lung_mask=True):
    if apply_lung_mask is True:
        image_masked = ma.masked_where(lung > 0.5, image)
        lung_mean = np.nanmean(image_masked)
        lung_std = np.nanstd(image_masked)
    else:
        lung_mean = np.nanmean(image)
        lung_std = np.nanstd(image)
    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


class CT_Dataset_Orthogonal(Dataset):
    '''
    Each volume should be at: Dimension X Height X Width
    '''
    def __init__(self,
                 imgs_folder,
                 labels_folder,
                 lung_folder,
                 labelled,
                 full_resolution=512,
                 lung_mask=False,
                 lung_window=True,
                 normalisation=True,
                 contrast_aug=True):
        # data
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.lung_folder = lung_folder
        # flags
        self.labelled_flag = labelled
        self.contrast_aug_flag = contrast_aug
        self.normalisation_flag = normalisation
        self.lung_window_flag = lung_window
        self.apply_lung_mask_flag = lung_mask

        if self.contrast_aug_flag is True:
            self.augmentation_contrast = RandomContrast([10, 255])

        self.augmentation_cropping = RandomCroppingOrthogonal(discarded_slices=1, resolution=full_resolution)

    def __getitem__(self, index):
        # Lung masks:
        all_lungs = sorted(glob.glob(os.path.join(self.lung_folder, '*.nii.gz*')))
        lung = nib.load(all_lungs[index])
        lung = lung.get_fdata()
        lung = np.array(lung, dtype='float32')
        lung = np.transpose(lung, (2, 0, 1))

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

        # Now applying lung window:
        if self.lung_window_flag is True:
            image[image < -1000.0] = -1000.0
            image[image > 500.0] = 500.0

        # Apply normalisation with values inside of lung
        if self.normalisation_flag is True:
            image = normalisation(lung, image, self.apply_lung_mask_flag)

        # Random contrast and Renormalisation:
        if self.contrast_aug_flag is True:
            image_another_contrast = self.augmentation_contrast.randomintensity(image)
            image = 0.7*image + 0.3*image_another_contrast
            if self.normalisation_flag is True:
                image = normalisation(lung, image, self.apply_lung_mask_flag)

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

            inputs_dict = self.augmentation_cropping.crop(image, label, lung)
            return inputs_dict, imagename
        else:
            inputs_dict = self.augmentation_cropping.crop(image, lung)
            return inputs_dict, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))


def getData(data_directory,
            dataset_name,
            train_batchsize,
            norm=True,
            contrast_aug=True,
            lung_window=True,
            resolution=512,
            train_full=True,
            unlabelled=False):
    '''
    Args:
        data_directory:
        dataset_name:
        train_batchsize:
        norm:
        contrast_aug:
        lung_window:
    Returns:
    '''

    # Labelled images data set and data loader:
    data_directory = data_directory + '/' + dataset_name
    train_image_folder_labelled = data_directory + '/labelled/imgs'
    train_label_folder_labelled = data_directory + '/labelled/lbls'
    train_lung_folder_labelled = data_directory + '/labelled/lung'

    train_dataset_labelled = CT_Dataset_Orthogonal(imgs_folder=train_image_folder_labelled,
                                                   labels_folder=train_label_folder_labelled,
                                                   lung_folder=train_lung_folder_labelled,
                                                   labelled=True,
                                                   full_resolution=resolution,
                                                   normalisation=norm,
                                                   contrast_aug=contrast_aug,
                                                   lung_window=lung_window)

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=True)

    # Unlabelled images data set and data loader:
    if unlabelled is True:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'
        train_label_folder_unlabelled = data_directory + '/unlabelled/lbls'
        train_lung_folder_unlabelled = data_directory + '/unlabelled/lung'

        train_dataset_unlabelled = CT_Dataset_Orthogonal(imgs_folder=train_image_folder_unlabelled,
                                                         labels_folder=train_label_folder_unlabelled,
                                                         lung_folder=train_lung_folder_unlabelled,
                                                         labelled=False,
                                                         full_resolution=resolution,
                                                         normalisation=norm,
                                                         contrast_aug=contrast_aug,
                                                         lung_window=lung_window)

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=train_batchsize,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

        if train_full is True:
            return {'train_data_l': train_dataset_labelled,
                    'train_loader_l': train_loader_labelled,
                    'train_data_u': train_dataset_unlabelled,
                    'train_loader_u': train_loader_unlabelled}
        else:
            validate_image_folder = data_directory + '/validate/imgs'
            validate_label_folder = data_directory + '/validate/lbls'
            validate_lung_folder = data_directory + '/validate/lung'

            validate_dataset = CT_Dataset_Orthogonal(imgs_folder=validate_image_folder,
                                                     labels_folder=validate_label_folder,
                                                     lung_folder=validate_lung_folder,
                                                     labelled=True,
                                                     full_resolution=resolution,
                                                     normalisation=norm,
                                                     contrast_aug=contrast_aug,
                                                     lung_window=lung_window)

            validate_loader = data.DataLoader(dataset=validate_dataset,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=0,
                                              drop_last=True)

            return {'train_data_l': train_dataset_labelled,
                    'train_loader_l': train_loader_labelled,
                    'train_data_u': train_dataset_unlabelled,
                    'train_loader_u': train_loader_unlabelled,
                    'val_data': validate_dataset,
                    'val_loader': validate_loader}

    else:
        if train_full is True:
            return {'train_data_l': train_dataset_labelled,
                    'train_loader_l': train_loader_labelled}
        else:
            validate_image_folder = data_directory + '/validate/imgs'
            validate_label_folder = data_directory + '/validate/lbls'
            validate_lung_folder = data_directory + '/validate/lung'

            validate_dataset = CT_Dataset_Orthogonal(imgs_folder=validate_image_folder,
                                                     labels_folder=validate_label_folder,
                                                     lung_folder=validate_lung_folder,
                                                     labelled=True,
                                                     full_resolution=resolution,
                                                     normalisation=norm,
                                                     contrast_aug=contrast_aug,
                                                     lung_window=lung_window)

            validate_loader = data.DataLoader(dataset=validate_dataset,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=0,
                                              drop_last=True)

            return {'train_data_l': train_dataset_labelled,
                    'train_loader_l': train_loader_labelled,
                    'val_data': validate_dataset,
                    'val_loader': validate_loader}


if __name__ == '__main__':
    dummy_input = np.random.rand(512, 512, 480)


