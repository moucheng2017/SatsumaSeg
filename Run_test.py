import torch
# sys.path.append("..")
from Train_Unet_simPL_EM import trainModels
# from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    trainModels(data_directory='/home/moucheng/projects_data',
                dataset_name='Task08_HepaticVessel',
                downsample=2,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                num_steps=4000,
                learning_rate=3e-2,
                width=16,
                log_tag='miccai',
                unlabelled=1,
                temperature=1.2,
                new_resolution=[8, 448, 448],
                l2=1e-4,
                alpha=0.05,
                warmup=0.5
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
