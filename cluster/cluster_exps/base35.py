import sys
import torch
sys.path.append("../..")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from TrainSup import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/private/',
                dataset_name='airway',
                repeat=1,
                train_batchsize=8,
                val_batchsize=0,
                num_steps=5000,
                learning_rate=1e-2,
                width=48,
                log_tag='2022_06_30',
                l2=1e-2,
                temp=2.0,
                new_d=1,
                new_h=480,
                new_w=480,
                resume_epoch=0,
                resume_training=False,
                checkpoint_path='/some/path')