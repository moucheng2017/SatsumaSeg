import sys
sys.path.append("..")
from Train_Unet_simPL import trainModels

if __name__ == '__main__':

    trainModels(data_directory='/SAN/medic/PerceptronHead/data/lung/public/',
                dataset_name='airway',
                dataset_tag='mismatch_exp',
                downsample=0,
                input_dim=1,
                class_no=2,
                repeat=3,
                train_batchsize=1,
                num_steps=1600,
                learning_rate=1e-3,
                width=16,
                log_tag='20220120',
                new_resolution=[16, 512, 512]
                )