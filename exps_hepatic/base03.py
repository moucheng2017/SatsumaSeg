import sys
import torch
sys.path.append("..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data',
                dataset_name='Task08_HepaticVessel',
                downsample=0,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=5000,
                learning_rate=3e-2,
                width=16,
                log_tag='miccai',
                new_resolution=[8, 448, 448],
                l2=1e-4
                )