import sys
import torch
sys.path.append("..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet_SegPL import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                downsample=4,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=5000,
                learning_rate=5e-3,
                width=16,
                unlabelled=2,
                log_tag='220200309',
                new_resolution=[16, 480, 480],
                l2=1e-4,
                alpha=1.0,
                warmup=0.5,
                threshold=0.4
                )