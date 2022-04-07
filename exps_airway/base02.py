import sys
import torch
sys.path.append("..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                downsample=2,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=4,
                num_steps=3000,
                learning_rate=1e-3,
                width=64,
                log_tag='220200309',
                new_resolution=[8, 480, 480],
                l2=1e-2
                )