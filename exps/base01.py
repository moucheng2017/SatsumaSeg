import sys
sys.path.append("..")
from Train_Unet import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/public/',
                dataset_name='airway',
                dataset_tag='512',
                downsample=0,
                input_dim=1,
                class_no=2,
                repeat=5,
                train_batchsize=1,
                num_steps=8000,
                learning_rate=1e-3,
                width=16,
                log_tag='20210125',
                new_resolution=[12, 512, 512],
                l2=0.001
                )