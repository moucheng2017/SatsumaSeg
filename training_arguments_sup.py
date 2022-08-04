import argparse

# dataset_name,
# data_directory,
# repeat,
# log_tag,
# num_steps=10000,
# learning_rate=1e-3,
# width=32,
# depth=4,
# train_batchsize=4,
# norm=False,
# contrast=False,
# lung=True,
# temp=2.0,
# l2=1e-3,
# resume_epoch=0,
# resume_training=False,
# checkpoint_path='/path/checkpoint/model'

parser = argparse.ArgumentParser(description='Training options for Satusma Seg for Pulmonary Tubular Structure Segmentation')

# paths to the training data
parser.add_argument('--data_path', type=str, help='path to the dataset, parent folder of the name of the dataset')
parser.add_argument('--dataset', type=str, help='name of the dataset')
parser.add_argument('--log_tag', type=str, help='experiment tag for the record')

# hyper parameters for training:
parser.add_argument('--repeat', type=int, help='number of times we repeat the experiment')
parser.add_argument('--iterations', type=int, help='number of iterations') #todo change the training script later
parser.add_argument('--lr', type=float, help='learning rate') #todo change the training scriot later
parser.add_argument('--width', type=int, help='number of filters in the first conv block in encoder')
parser.add_argument('--depth', type=int, help='number of downsampling stages')
parser.add_argument('--batch', type=int, help='number of training batch size')

# flags for data preprocessing and augmentation in data loader:
parser.add_argument('--norm', type=bool, help='true when using z score normalisation on images')
parser.add_argument('--contrast', type=bool, help='random contrast augmentation using histogram equalization')
parser.add_argument('--lung', type=bool, help='true when applying lung mask on the training data')
# parser.add_argument('--temp', '--t', type=)
