import torch
# sys.path.append("..")
from Train_Unet_Orthogonal import trainModels
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    # Orthogonal planes:
    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
                dataset_name='airway',
                repeat=1,
                train_batchsize=4,
                num_steps=100000,
                learning_rate=0.01,
                width=16,
                log_tag='2022_07_15',
                l2=1e-3,
                temp=1.0,
                norm=False,
                contrast=False,
                lung=True,
                resume_epoch=0,
                resume_training=False,
                checkpoint_path='/some/path')

    # with torch.autograd.set_detect_anomaly(True):
    # SegPL-VI:
    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
    #             dataset_name='airway',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=3,
    #             num_steps=3000,
    #             learning_rate=1e-3,
    #             width=24,
    #             log_tag='airway_balanced',
    #             unlabelled=1,
    #             new_resolution=[1, 480, 480],
    #             l2=1e-2,
    #             alpha=1.0,
    #             warmup=1.0,
    #             mean=0.4,
    #             std=0.12
    #             )

    # 2D
    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data',
    #             dataset_name='airway',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=2,
    #             num_steps=5000,
    #             learning_rate=1e-3,
    #             width=64,
    #             log_tag='airway_balanced',
    #             new_resolution=[16, 480, 480],
    #             l2=1e-2
    #             )


