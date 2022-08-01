import sys
import torch
sys.path.append("../..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Train_Unet import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                downsample=0,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=1200,
                learning_rate=1e-3,
                width=64,
                log_tag='22020411',
                new_resolution=[3, 480, 480],
                l2=1e-2
                )