import sys
import torch
sys.path.append("..")

# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                dataset_tag='Mixed',
                downsample=4,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=4000,
                learning_rate=1e-4,
                width=16,
                log_tag='20200206',
                new_resolution=[16, 384, 384],
                l2=5e-2
                )