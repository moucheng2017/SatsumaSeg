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


class HipCTDataset(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder=None,
                 input_shape=(150, 150, 150),
                 output_shape=(160, 160, 160),
                 gaussian_aug=1,
                 zoom_aug=1,
                 contrast_aug=1):

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

        self.augmentation_cropping = RandomSlicingOrthogonalFast(discarded_slices=1,
                                                                 zoom=zoom_aug,
                                                                 input_size=input_shape,
                                                                 output_size=output_shape)

    def __getitem__(self, index):
        # Check image extension:
        all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.npy*')))
        imagename = all_images[index]
        # load image and preprocessing:
        image = np.load(imagename)

        image = np.array(image, dtype='float32')
        # transform dimension:
        image = np.transpose(image, (2, 0, 1)) # (H x W x D) --> (D x H x W)

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        if self.lbls_folder:
            # Labels:
            all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.npy*')))
            label = np.load(all_labels[index])

            label = np.array(label, dtype='float32')
            label = np.transpose(label, (2, 0, 1))

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
            image_weighted = normalisation(label, image_weighted)

            # get slices by weighted sampling on each axis with zoom in augmentation:
            inputs_dict = self.augmentation_cropping.crop(image_weighted, label)

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
        return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))


class CT_Dataset_Orthogonal(Dataset):
    '''
    Each volume should be at: Dimension X Height X Width
    Sequentially random augment image with multiple steps
    '''
    def __init__(self,
                 images_folder,
                 labels_folder=None,
                 new_size_h=384,
                 new_size_w=384,
                 full_orthogonal=0,
                 sampling_weight=5,
                 lung_window=1,
                 normalisation=1,
                 gaussian_aug=1,
                 zoom_aug=1,
                 contrast_aug=1):

        # flags
        # self.labelled_flag = labelled
        self.contrast_aug_flag = contrast_aug
        self.gaussian_aug_flag = gaussian_aug
        self.normalisation_flag = normalisation
        self.zoom_aug_flag = zoom_aug

        # data
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        self.lung_window_flag = lung_window

        if self.contrast_aug_flag == 1:
            self.augmentation_contrast = RandomContrast(bin_range=(20, 255))

        if self.gaussian_aug_flag == 1:
            self.gaussian_noise = RandomGaussian()

        self.augmentation_cropping = RandomSlicingOrthogonal(discarded_slices=1,
                                                             zoom=zoom_aug,
                                                             sampling_weighting_slope=sampling_weight,
                                                             full_orthogonal=full_orthogonal,
                                                             new_size_w=new_size_w,
                                                             new_size_h=new_size_h)

    def __getitem__(self, index):
        # Lung masks:
        # all_lungs = sorted(glob.glob(os.path.join(self.lung_folder, '*.nii.gz*')))
        # lung = nib.load(all_lungs[index])
        # lung = lung.get_fdata()
        # lung = np.array(lung, dtype='float32')
        # lung = np.transpose(lung, (2, 0, 1))

        # Check image extension:
        image_example = os.listdir(self.imgs_folder)[0]
        if image_example.lower().endswith(('.nii.gz', '.nii')):
            # Images:
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
            imagename = all_images[index]
            # load image and preprocessing:
            image = nib.load(imagename)
            image = image.get_fdata()

        else:
            # Images:
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.npy*')))
            imagename = all_images[index]
            # load image and preprocessing:
            image = np.load(imagename)

        image = np.array(image, dtype='float32')
        # transform dimension:
        image = np.transpose(image, (2, 0, 1)) # (H x W x D) --> (D x H x W)

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        # Now applying lung window:
        if self.lung_window_flag == 1:
            image[image < -1000.0] = -1000.0
            image[image > 500.0] = 500.0

        if self.lbls_folder:
            # Labels:
            label_example = os.listdir(self.lbls_folder)[0]
            if label_example.lower().endswith(('.nii.gz', '.nii')):
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
                label = nib.load(all_labels[index])
                label = label.get_fdata()
            else:
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.npy*')))
                label = np.load(all_labels[index])

            label = np.array(label, dtype='float32')
            label = np.transpose(label, (2, 0, 1))

            image_queue = collections.deque()

            # Apply normalisation at each case-wise:
            # if self.normalisation_flag == 1:
            #     image = normalisation(label, image)

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
            if self.normalisation_flag == 1:
                image_weighted = normalisation(label, image_weighted)

            # get slices by weighted sampling on each axis with zoom in augmentation:
            inputs_dict = self.augmentation_cropping.crop(image_weighted, label)

            return inputs_dict, imagename

        else:
            image_queue = collections.deque()

            # Apply normalisation at each case-wise:
            # if self.normalisation_flag == 1:
                # image = normalisation(None, image)

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
            if self.normalisation_flag == 1:
                image_weighted = normalisation(None, image_weighted)

            inputs_dict = self.augmentation_cropping.crop(image_weighted)

            return inputs_dict, imagename

    def __len__(self):
        example = os.listdir(self.imgs_folder)[0]
        if example.lower().endswith(('.nii.gz', '.nii')):
            # You should change 0 to the total size of your dataset.
            return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
        elif example.lower().endswith('.npy'):
            return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))


def getData(data_directory,
            train_batchsize,
            sampling_weight,
            new_size_h,
            new_size_w,
            full_sampling_mode=0,
            norm=1,
            zoom_aug=1,
            contrast_aug=1,
            lung_window=1,
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

    train_image_folder_labelled = data_directory + '/labelled/imgs'
    train_label_folder_labelled = data_directory + '/labelled/lbls'

    train_dataset_labelled = CT_Dataset_Orthogonal(images_folder=train_image_folder_labelled,
                                                   labels_folder=train_label_folder_labelled,
                                                   sampling_weight=sampling_weight,
                                                   new_size_h=new_size_h,
                                                   new_size_w=new_size_w,
                                                   normalisation=norm,
                                                   zoom_aug=zoom_aug,
                                                   contrast_aug=contrast_aug,
                                                   lung_window=lung_window,
                                                   full_orthogonal=full_sampling_mode)

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=True)

    # Unlabelled images data set and data loader:
    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'

        train_dataset_unlabelled = CT_Dataset_Orthogonal(images_folder=train_image_folder_unlabelled,
                                                         sampling_weight=sampling_weight,
                                                         zoom_aug=0,
                                                         normalisation=norm,
                                                         new_size_h=new_size_h,
                                                         new_size_w=new_size_w,
                                                         contrast_aug=contrast_aug,
                                                         lung_window=lung_window,
                                                         full_orthogonal=full_sampling_mode)

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=train_batchsize*unlabelled,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled,
                'train_data_u': train_dataset_unlabelled,
                'train_loader_u': train_loader_unlabelled}

    else:
        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled}


def getHipData(data_directory,
               train_batchsize=1,
               zoom_aug=1,
               contrast_aug=1,
               unlabelled=1):
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

    train_dataset_labelled = HipCTDataset(images_folder=train_image_folder_labelled,
                                          labels_folder=train_label_folder_labelled,
                                          zoom_aug=zoom_aug,
                                          contrast_aug=contrast_aug)

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=True)

    # Unlabelled images data set and data loader:
    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'

        train_dataset_unlabelled = HipCTDataset(images_folder=train_image_folder_unlabelled,
                                                zoom_aug=0,
                                                contrast_aug=contrast_aug)

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=train_batchsize*unlabelled,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled,
                'train_data_u': train_dataset_unlabelled,
                'train_loader_u': train_loader_unlabelled}

    else:
        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled}


if __name__ == '__main__':
    dummy_input = np.random.rand(512, 512, 480)


