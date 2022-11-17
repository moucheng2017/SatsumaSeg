import nibabel as nib
import collections
import glob
import os
import numpy.ma as ma
from libs.Augmentations import *
from torch.utils import data
from torch.utils.data import Dataset


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


class CustomDataset(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder=None,
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

        # data
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        if self.contrast_aug_flag == 1:
            self.augmentation_contrast = RandomContrast(bin_range=(20, 255))

        if self.gaussian_aug_flag == 1:
            self.gaussian_noise = RandomGaussian()

        if full_orthogonal == 1:
            self.augmentation_cropping = RandomSlicingOrthogonal(zoom=zoom_aug,
                                                                 output_size=output_shape,
                                                                 full_orthogonal=full_orthogonal)

        else:
            self.augmentation_cropping = RandomSlicingOrthogonal(zoom=zoom_aug,
                                                                 output_size=output_shape,
                                                                 full_orthogonal=full_orthogonal)

    def __getitem__(self, index):
        # Check image extension:
        # all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.npy*')))
        all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
        imagename = all_images[index]
        # load image and preprocessing:
        image = nib.load(imagename)
        image = image.get_fdata()

        # imagename = all_images[index]
        # load image and preprocessing:
        # image = np.load(imagename)

        image = np.array(image, dtype='float32')
        # transform dimension:
        # image = np.transpose(image, (2, 0, 1)) # (H x W x D) --> (D x H x W)

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        if self.lbls_folder:
            # Labels:
            all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
            label = nib.load(all_labels[index])
            label = label.get_fdata()

            label = np.array(label, dtype='float32')
            # label = np.transpose(label, (2, 0, 1))

            image_queue = collections.deque()

            image_queue.append(image)

            # Random contrast:
            if self.contrast_aug_flag == 1:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image_queue.append(image_another_contrast)

            # Random Gaussian:
            if self.gaussian_aug_flag == 1:
                image_noise = self.gaussian_noise.gaussiannoise(image)
                image_queue.append(image_noise)

            # weights:
            dirichlet_alpha = collections.deque()
            for i in range(len(image_queue)):
                dirichlet_alpha.append(1)
            dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)

            # make a new image:
            image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
            image_weighted = sum(image_weighted)

            # average sum:
            # image_weighted = sum(image_queue) / len(image_queue)

            # Apply normalisation at each case-wise again:
            image_weighted = normalisation(label, image_weighted)

            # get slices by weighted sampling on each axis with zoom in augmentation:
            inputs_dict = self.augmentation_cropping.crop(image_weighted,
                                                          label)

            return inputs_dict, imagename

        else:
            image_queue = collections.deque()
            image_queue.append(image)

            # Random contrast:
            if self.contrast_aug_flag == 1:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image_queue.append(image_another_contrast)

            # Random Gaussian:
            if self.gaussian_aug_flag == 1:
                image_noise = self.gaussian_noise.gaussiannoise(image)
                image_queue.append(image_noise)

            # weights:
            dirichlet_alpha = collections.deque()
            for i in range(len(image_queue)):
                dirichlet_alpha.append(1)
            dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)

            # make a new image:
            image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
            image_weighted = sum(image_weighted)

            # Apply normalisation at each case-wise again:
            image_weighted = normalisation(None, image_weighted)

            inputs_dict = self.augmentation_cropping.crop(image_weighted)

            return inputs_dict, imagename

    def __len__(self):
        return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz')))
        # return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))


def getData(data_directory,
            train_batchsize=1,
            zoom_aug=1,
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

    val_image_folder_labelled = data_directory + '/test/imgs'
    val_label_folder_labelled = data_directory + '/test/lbls'

    val_dataset_labelled = CustomDataset(images_folder=val_image_folder_labelled,
                                         labels_folder=val_label_folder_labelled,
                                         zoom_aug=0,
                                         contrast_aug=0,
                                         output_shape=output_shape,
                                         full_orthogonal=full_orthogonal,
                                         gaussian_aug=0
                                         )

    val_loader_labelled = data.DataLoader(dataset=val_dataset_labelled,
                                          batch_size=1,
                                          shuffle=True,
                                          drop_last=True)

    # Unlabelled images data set and data loader:
    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'

        train_dataset_unlabelled = CustomDataset(images_folder=train_image_folder_unlabelled,
                                                 zoom_aug=0,
                                                 contrast_aug=contrast_aug,
                                                 output_shape=output_shape,
                                                 full_orthogonal=full_orthogonal,
                                                 gaussian_aug=gaussian_aug
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




