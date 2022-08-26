import glob
import os
import torch
import random


import numpy as np
import scipy.ndimage
import nibabel as nib
import numpy.ma as ma

from torch.utils import data
from torch.utils.data import Dataset

from libs.Augmentations import *


def normalisation(label, image):
    # Case-wise normalisation
    # Normalisation using values inside of the foreground mask

    if label is None:
        lung_mean = np.nanmean(image)
        lung_std = np.nanstd(image)
    else:
        image_masked = ma.masked_where(label > 0.5, image)
        lung_mean = np.nanmean(image_masked)
        lung_std = np.nanstd(image_masked)

    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


class CT_Dataset_Orthogonal(Dataset):
    '''
    Each volume should be at: Dimension X Height X Width
    '''
    def __init__(self,
                 imgs_folder,
                 labels_folder,
                 labelled,
                 full_resolution=512,
                 sampling_weight=10,
                 lung_window=True,
                 normalisation=True,
                 gaussian_aug=True,
                 zoom_aug=True,
                 contrast_aug=True):
        # data
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder

        # flags
        self.labelled_flag = labelled
        self.contrast_aug_flag = contrast_aug
        self.gaussian_aug_flag = gaussian_aug
        self.normalisation_flag = normalisation
        self.zoom_aug_flag = zoom_aug

        # we now removed lung masking
        # self.lung_folder = lung_folder
        # self.apply_lung_mask_flag = lung_mask

        self.lung_window_flag = lung_window

        if self.contrast_aug_flag is True:
            self.augmentation_contrast = RandomContrast([10, 255])

        if self.gaussian_aug_flag is True:
            self.gaussian_noise = RandomGaussian()

        self.augmentation_cropping = RandomCroppingOrthogonal(discarded_slices=1,
                                                              zoom=zoom_aug,
                                                              resolution=full_resolution,
                                                              sampling_weighting_slope=sampling_weight)

    def __getitem__(self, index):
        # Lung masks:
        # all_lungs = sorted(glob.glob(os.path.join(self.lung_folder, '*.nii.gz*')))
        # lung = nib.load(all_lungs[index])
        # lung = lung.get_fdata()
        # lung = np.array(lung, dtype='float32')
        # lung = np.transpose(lung, (2, 0, 1))

        # Images:
        all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
        imagename = all_images[index]

        # load image and preprocessing:
        image = nib.load(imagename)
        image = image.get_fdata()
        image = np.array(image, dtype='float32')

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        # transform dimension:
        # original dimension: (H x W x D)
        image = np.transpose(image, (2, 0, 1))

        # Now applying lung window:
        if self.lung_window_flag is True:
            image[image < -1000.0] = -1000.0
            image[image > 500.0] = 500.0

        # Random contrast:
        if self.contrast_aug_flag is True:
            if random.random() > 0.5:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image = image_another_contrast

        # Random Gaussian:
        if self.gaussian_aug_flag is True:
            if random.random() > 0.5:
                image = self.gaussian_noise.gaussiannoise(image)

        if self.labelled_flag is True:
            # Labels:
            all_labels = sorted(glob.glob(os.path.join(self.labels_folder, '*.nii.gz*')))
            label = nib.load(all_labels[index])
            label = label.get_fdata()
            label = np.array(label, dtype='float32')
            label = np.transpose(label, (2, 0, 1))

            # Apply normalisation at each case-wise:
            if self.normalisation_flag is True:
                image = normalisation(label, image)

            # get slices by weighted sampling on each axis with zoom in augmentation:
            inputs_dict = self.augmentation_cropping.crop(image, label)

            return inputs_dict, imagename

        else:
            if self.normalisation_flag is True:
                image = normalisation(None, image)

            inputs_dict = self.augmentation_cropping.crop(image)

            return inputs_dict, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))


def getData(data_directory,
            dataset_name,
            train_batchsize,
            sampling_weight,
            norm=True,
            zoom_aug=True,
            contrast_aug=True,
            lung_window=True,
            resolution=512,
            train_full=True,
            unlabelled=2):
    '''
    Args:
        data_directory:
        dataset_name:
        train_batchsize:
        norm:
        contrast_aug:
        lung_window:
        resolution:
        train_full:
        unlabelled:

    Returns:

    '''
    # Labelled images data set and data loader:
    data_directory = data_directory + '/' + dataset_name
    train_image_folder_labelled = data_directory + '/labelled/imgs'
    train_label_folder_labelled = data_directory + '/labelled/lbls'
    # train_lung_folder_labelled = data_directory + '/labelled/lung'

    train_dataset_labelled = CT_Dataset_Orthogonal(imgs_folder=train_image_folder_labelled,
                                                   labels_folder=train_label_folder_labelled,
                                                   labelled=True,
                                                   sampling_weight=sampling_weight,
                                                   full_resolution=resolution,
                                                   normalisation=norm,
                                                   zoom_aug=zoom_aug,
                                                   contrast_aug=contrast_aug,
                                                   lung_window=lung_window)

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=True)

    # Unlabelled images data set and data loader:
    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'
        train_label_folder_unlabelled = data_directory + '/unlabelled/lbls'
        # train_lung_folder_unlabelled = data_directory + '/unlabelled/lung'

        train_dataset_unlabelled = CT_Dataset_Orthogonal(imgs_folder=train_image_folder_unlabelled,
                                                         labels_folder=train_label_folder_unlabelled,
                                                         labelled=False,
                                                         sampling_weight=sampling_weight,
                                                         zoom_aug=False,
                                                         full_resolution=resolution,
                                                         normalisation=norm,
                                                         contrast_aug=contrast_aug,
                                                         lung_window=lung_window)

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=train_batchsize*unlabelled,
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
            # validate_lung_folder = data_directory + '/validate/lung'

            validate_dataset = CT_Dataset_Orthogonal(imgs_folder=validate_image_folder,
                                                     labels_folder=validate_label_folder,
                                                     sampling_weight=sampling_weight,
                                                     zoom_aug=False,
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
            # validate_lung_folder = data_directory + '/validate/lung'

            validate_dataset = CT_Dataset_Orthogonal(imgs_folder=validate_image_folder,
                                                     labels_folder=validate_label_folder,
                                                     sampling_weight=sampling_weight,
                                                     zoom_aug=False,
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


