import sys
import torch
sys.path.append("..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet_Orthogonal_Single import trainModels

if __name__ == '__main__':

    # trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
    #             dataset_name='airway',
    #             repeat=1,
    #             train_batchsize=5,
    #             val_batchsize=5,
    #             num_steps=4000,
    #             learning_rate=1e-3,
    #             width=48,
    #             log_tag='2022_06_11',
    #             l2=0.01,
    #             temp=1.0,
    #             new_d=5,
    #             new_h=448,
    #             new_w=448,
    #             new_z=320)

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                repeat=1,
                train_batchsize=6,
                val_batchsize=0,
                num_steps=4000,
                learning_rate=1e-2,
                width=64,
                log_tag='2022_06_23',
                l2=1e-2,
                temp=2.0,
                new_d=15,
                new_h=448,
                new_w=448,
                resume_epoch=0,
                resume_training=False,
                checkpoint_path='/some/path')