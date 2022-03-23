import torch
# sys.path.append("..")
# from Train_Unet_SegPLVI import trainModels
from Train_Unet import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # SegPL-VI:
    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
    #             dataset_name='airway',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=1,
    #             num_steps=2000,
    #             learning_rate=1e-4,
    #             width=32,
    #             log_tag='airway_segpl_vi_pat_2d',
    #             unlabelled=2,
    #             new_resolution=[1, 480, 480],
    #             l2=1e-4,
    #             alpha=0.1,
    #             warmup=1.0,
    #             mean=0.4,
    #             std=0.12
    #             )

    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
                dataset_name='airway',
                downsample=4,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=12,
                num_steps=2000,
                learning_rate=1e-5,
                width=32,
                log_tag='airway_pat',
                new_resolution=[1, 480, 480],
                l2=1e-2
                )
