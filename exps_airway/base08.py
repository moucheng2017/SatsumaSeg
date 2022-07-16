import sys
import torch
sys.path.append("..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from arxiv.Train_Unet_Orthogonal import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                input_dim=5,
                repeat=1,
                train_batchsize=6,
                num_steps=5000,
                learning_rate=1e-3,
                width=64,
                log_tag='2022_05_13',
                l2=1e-2,
                temp=2.0
                )