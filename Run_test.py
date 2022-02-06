import torch
# sys.path.append("..")
# from Train_Unet_simPL import trainModels
from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
    #             dataset_name='airway',
    #             dataset_tag='Mixed',
    #             downsample=4,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=1,
    #             num_steps=800,
    #             learning_rate=1e-4,
    #             width=16,
    #             log_tag='20220202',
    #             new_resolution=[16, 256, 256],
    #             l2=1e-4,
    #             alpha=1.0,
    #             warmup=1.0
    #             )

    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
                dataset_name='airway',
                dataset_tag='Mixed',
                downsample=4,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=1000,
                learning_rate=1e-4,
                width=16,
                log_tag='testing',
                new_resolution=[16, 256, 256],
                l2=5e-2
                )
