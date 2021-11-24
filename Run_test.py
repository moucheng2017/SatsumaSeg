import torch
import sys
# sys.path.append("..")

from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':
    # How to use this:
    # 1. data_directory, you can try out the default one with example images
    # 2. dataset_name. In the example datasets, we have two examples: binary and multiclass
    # 3. input_dim. In the example datasets, if you select binary, then change input_dim to 3 for RGB
    #    images from CityScapes; if you select multiclass, then change input_dim to 4,
    #    those images are 4D examples from BRATS
    # 4. class_no. If you select binary, set up this to 2; if you select multiclass, set up this to 4
    # 5. repeat. Number of running times of the same experiments, to test performance variance.
    # 6. network. We provide a range of candidate networks: 'unet', 'dilated_unet', 'fcn', 'deeper_unet',
    # 'atten_unet', 'cse_unet_full'.
    # 7. width. number of channels in the first encoder in the network:
    # 8. augmentation. parameter to choose different data augmentation. We have options: 'full', 'flip',
    # 'all_filp', 'gaussian' or 'none'.
    # 9. The results will be genereted in a new folder called Results.

    # trainModels(data_directory='./datasets/',
    #             dataset_name='multiclass',
    #             input_dim=4,
    #             class_no=4,
    #             repeat=1,
    #             train_batchsize=4,
    #             validate_batchsize=1,
    #             num_epochs=1,
    #             learning_rate=1e-3,
    #             width=64,
    #             network='unet',
    #             augmentation='full'
    #             )

    trainModels(data_directory='/cluster/project2/Neuroblastoma/',
                dataset_name='data',
                input_dim=3,
                class_no=2,
                repeat=1,
                train_batchsize=2,
                validate_batchsize=1,
                num_epochs=100,
                learning_rate=1e-3,
                width=32,
                network='unet',
                augmentation='full'
                )


