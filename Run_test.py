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
                downsample=0,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=2000,
                learning_rate=0.02,
                width=8,
                log_tag='airway_segpl_vi',
                unlabelled=2,
                new_resolution=[3, 224, 224],
                l2=1e-5,
                alpha=100.0,
                warmup=0.5,
                mean=0.4,
                std=0.12
                )

    # trainModels(data_directory='/home/moucheng/projects_data',
    #             dataset_name='Task08_HepaticVessel',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             num_steps=200,
    #             learning_rate=1e-2,
    #             width=8,
    #             log_tag='miccailung',
    #             new_resolution=[16, 448, 448],
    #             l2=1e-4
    #             )
