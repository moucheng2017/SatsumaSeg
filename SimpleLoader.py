import torchvision.transforms.functional as TF
from torch.utils import data
from torch.utils.data import Dataset
import random
import nibabel as nib
import torchvision.transforms as transforms
import numpy as np
import glob
from torch.utils.data import DataLoader
import os


class SegDataset(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder,
                 output_shape=(150, 150)):

        self.images_folder = images_folder
        self.labels_folder = labels_folder

        self.h = output_shape[0]
        self.w = output_shape[1]

    def __getitem__(self, index):

        all_images = sorted(glob.glob(os.path.join(self.images_folder, '*.nii.gz*')))
        all_labels = sorted(glob.glob(os.path.join(self.labels_folder, '*.nii.gz*')))

        image_name = all_images[index]
        label_name = all_labels[index]

        image = nib.load(image_name)
        image = image.get_fdata()

        label = nib.load(label_name)
        label = label.get_fdata()

        # Slicing:
        h, w, d = np.shape(image)
        image = image[:, :, d // 2]  # let's sample a slice in the middle of the volume
        label = label[:, :, d // 2]  # let's sample a slice in the middle of the volume
        image = transforms.ToPILImage()(image)
        label = transforms.ToPILImage()(label)

        # Resize
        resize_img = transforms.Resize(size=(self.h, self.w), interpolation=transforms.InterpolationMode.BILINEAR)
        resize_lbl = transforms.Resize(size=(self.h, self.w), interpolation=transforms.InterpolationMode.NEAREST)
        image = resize_img(image)
        label = resize_lbl(label)

        # Random crop
        new_h, new_w = np.random.randint(self.h // 2, self.h), np.random.randint(self.w // 2, self.w)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(new_h, new_w))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # Resize again:
        image = resize_img(image)
        label = resize_lbl(label)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Random contrast with hist equalisation
        image = transforms.RandomEqualize(0.5)(image)

        return image, label

    def __len__(self):
        return len(glob.glob(os.path.join(self.images_folder, '*.nii.gz*')))


if __name__ == '__main__':

    train_data = SegDataset('/home/moucheng/projects_data/Pulmonary_data/airway/labelled/imgs',
                            '/home/moucheng/projects_data/Pulmonary_data/airway/labelled/lbls',
                            (140, 140))
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
    for batch_idx, data in enumerate(train_loader, 0):
        x, y = data
        break
