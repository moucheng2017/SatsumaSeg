import torch
# sys.path.append("..")
from Train_Unet_simPL_Fixed import trainModels
# from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # simPL:
    trainModels(data_directory='/home/moucheng/projects_data',
                dataset_name='Task08_HepaticVessel',
                downsample=3,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=200,
                learning_rate=1e-2,
                width=8,
                log_tag='miccai',
                unlabelled=3,
                new_resolution=[16, 320, 320],
                l2=1e-4,
                alpha=1.0,
                warmup=0.8
                )

    # trainModels(data_directory='/home/moucheng/projects_data',
    #             dataset_name='Task08_HepaticVessel',
    #             downsample=3,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=2,
    #             num_steps=200,
    #             learning_rate=1e-2,
    #             width=8,
    #             log_tag='miccai',
    #             new_resolution=[16, 480, 480],
    #             l2=1e-4
    #             )
