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

    # trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
    #             dataset_name='airway',
    #             dataset_tag='mismatch_exp',
    #             downsample=3,
    #             input_dim=1,
    #             class_no=2,
    #             repeat=3,
    #             train_batchsize=1,
    #             num_steps=100,
    #             learning_rate=2e-4,
    #             width=16,
    #             log_tag='test_random_crop',
    #             new_resolution=192,
    #             spatial_consistency='global_local'
    #             )

    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
                dataset_name='airway',
                dataset_tag='mismatch_exp',
                downsample=3,
                input_dim=1,
                class_no=2,
                repeat=3,
                train_batchsize=1,
                num_steps=100,
                learning_rate=2e-4,
                width=16,
                log_tag='test_random_crop',
                new_resolution=192,
                spatial_consistency='none'
                )
