import torch
# sys.path.append("..")
from Train_Unet_SegPLVI import trainModels
# from Train_Unet import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # SegPL-VI:
    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
                dataset_name='airway',
                downsample=4,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=5,
                num_steps=3000,
                learning_rate=1e-3,
                width=64,
                log_tag='airway_balanced',
                unlabelled=2,
                new_resolution=[1, 480, 480],
                l2=1e-2,
                alpha=1.0,
                warmup=1.0,
                mean=0.4,
                std=0.12
                )

    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
    #             dataset_name='airway',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=5,
    #             num_steps=2000,
    #             learning_rate=1e-3,
    #             width=64,
    #             log_tag='airway_balanced',
    #             new_resolution=[1, 480, 480],
    #             l2=1e-2
    #             )
