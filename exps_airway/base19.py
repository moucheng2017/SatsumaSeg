import sys
import torch
sys.path.append("..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet_Orthogonal import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                repeat=1,
                train_batchsize=6,
                val_batchsize=5,
                num_steps=5000,
                learning_rate=1e-3,
                width=48,
                log_tag='2022_05_31',
                l2=0.01,
                temp=0.5,
                new_d=5,
                new_h=448,
                new_w=448)