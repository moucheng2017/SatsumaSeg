import torch
# sys.path.append("..")
from Train_Unet_simPL import trainModels
# from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
                dataset_name='airway',
                dataset_tag='mismatch_exp',
                downsample=0,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=1,
                num_steps=8000,
                learning_rate=1e-3,
                width=12,
                log_tag='20220120_augmentation',
                new_resolution=[3, 400, 400]
                )

    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
    #             dataset_name='airway',
    #             dataset_tag='mismatch_exp',
    #             downsample=2,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=1,
    #             num_steps=24000,
    #             learning_rate=1e-3,
    #             width=16,
    #             log_tag='20220120_no_data_augmentation',
    #             new_resolution=[8, 224, 224]
    #             )
