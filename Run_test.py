import torch
# sys.path.append("..")

from Train_Unet_simPL import trainModels
# from Train_Unet import trainModels
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':

    trainModels(data_directory='/home/moucheng/projects_data/Pulmonary_data/',
                dataset_name='airway',
                dataset_tag='mismatch_exp',
                downsample=3,
                input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=1,
                num_steps=200,
                learning_rate=1e-4,
                width=16,
                log_tag='random_contrast_augmentation',
                new_resolution=160
                )
