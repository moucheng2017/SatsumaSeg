import torch
# sys.path.append("..")
from Train_Unet_simPL_Soft import trainModels
# from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    trainModels(data_directory='/home/moucheng/projects_data',
                dataset_name='Task06_Lung',
                downsample=3,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=12,
                learning_rate=1e-4,
                width=16,
                log_tag='20220218',
                new_resolution=[16, 256, 256],
                l2=1e-4,
                alpha=1.0,
                warmup=1.0
                )

    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
    #             dataset_name='airway',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=2,
    #             num_steps=1000,
    #             learning_rate=1e-4,
    #             width=16,
    #             log_tag='train_on_turkish',
    #             new_resolution=[32, 256, 256],
    #             l2=1e-2
    #             )
