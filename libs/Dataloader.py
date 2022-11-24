import random

import nibabel as nib
import collections
import glob
import os
import numpy.ma as ma
from libs.Augmentations import *
from torch.utils import data
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder=None,
                 data_format='np', # np for numpy and the default is nii
                 output_shape=(160, 160),
                 full_orthogonal=0,
                 gaussian_aug=1,
                 zoom_aug=1,
                 contrast_aug=1
                 ):

        # flags
        # self.labelled_flag = labelled
        self.contrast_aug_flag = contrast_aug
        self.gaussian_aug_flag = gaussian_aug
        self.zoom_aug_flag = zoom_aug
        self.data_format = data_format

        # data
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        if self.contrast_aug_flag == 1:
            self.augmentation_contrast = RandomContrast(bin_range=(20, 255))

        if self.gaussian_aug_flag == 1:
            self.gaussian_noise = RandomGaussian()

        self.augmentation_cropping = RandomSlicingOrthogonal(zoom=zoom_aug,
                                                             output_size=output_shape,
                                                             full_orthogonal=full_orthogonal)

    def __getitem__(self, index):
        # Check image extension:
        if self.data_format == 'np':
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.npy*')))
            imagename = all_images[index]
            image = np.load(imagename)
        else:
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
            imagename = all_images[index]
            image = nib.load(imagename)
            image = image.get_fdata()

        image = np.array(image, dtype='float32')

        # normalisation:
        image = norm95(image)

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        # Random Gaussian:
        if self.gaussian_aug_flag == 1:
            if random.random() > .5:
                image = self.gaussian_noise.gaussiannoise(image)

        # Random contrast:
        if self.contrast_aug_flag == 1:
            if random.random() > .5:
                image = self.augmentation_contrast.randomintensity(image)

        # Renormalisation:
        image = norm95(image)

        if self.lbls_folder:
            # Labels:
            if self.data_format == 'np':
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.npy')))
                label = np.load(all_labels[index])
            else:
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
                label = nib.load(all_labels[index])
                label = label.get_fdata()
            label = np.array(label, dtype='float32')
            inputs_dict = self.augmentation_cropping.crop(image,
                                                          label)
        else:
            inputs_dict = self.augmentation_cropping.crop(image)

        return inputs_dict, imagename

    def __len__(self):
        if self.data_format == 'np':
            return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))
        else:
            return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz')))


def getData(data_directory,
            train_batchsize=1,
            zoom_aug=1,
            data_format='np',
            contrast_aug=1,
            unlabelled=1,
            output_shape=(160, 160),
            full_orthogonal=0,
            gaussian_aug=1,
            ):
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

    train_image_folder_labelled = data_directory + '/labelled/imgs'
    train_label_folder_labelled = data_directory + '/labelled/lbls'

    train_dataset_labelled = CustomDataset(images_folder=train_image_folder_labelled,
                                           labels_folder=train_label_folder_labelled,
                                           zoom_aug=zoom_aug,
                                           data_format=data_format,
                                           contrast_aug=contrast_aug,
                                           output_shape=output_shape,
                                           full_orthogonal=full_orthogonal,
                                           gaussian_aug=gaussian_aug
                                           )

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True)

    val_image_folder_labelled = data_directory + '/validate/imgs'
    val_label_folder_labelled = data_directory + '/validate/lbls'

    val_dataset_labelled = CustomDataset(images_folder=val_image_folder_labelled,
                                         labels_folder=val_label_folder_labelled,
                                         zoom_aug=0,
                                         data_format=data_format,
                                         contrast_aug=0,
                                         output_shape=output_shape,
                                         full_orthogonal=full_orthogonal,
                                         gaussian_aug=0
                                         )

    val_loader_labelled = data.DataLoader(dataset=val_dataset_labelled,
                                          batch_size=1,
                                          shuffle=True,
                                          drop_last=False)

    # Unlabelled images data set and data loader:
    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'

        train_dataset_unlabelled = CustomDataset(images_folder=train_image_folder_unlabelled,
                                                 zoom_aug=0,
                                                 contrast_aug=0,
                                                 output_shape=output_shape,
                                                 data_format=data_format,
                                                 full_orthogonal=full_orthogonal,
                                                 gaussian_aug=0
                                                 )

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=train_batchsize*unlabelled,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  drop_last=True)

        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled,
                'val_data_l': val_dataset_labelled,
                'val_loader_l': val_loader_labelled,
                'train_data_u': train_dataset_unlabelled,
                'train_loader_u': train_loader_unlabelled}

    else:
        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled,
                'val_data_l': val_dataset_labelled,
                'val_loader_l': val_loader_labelled
                }




