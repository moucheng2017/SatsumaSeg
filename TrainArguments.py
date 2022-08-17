import argparse

parser = argparse.ArgumentParser(description='Training options for Satusma Seg for Pulmonary Tubular Structure Segmentation, both supervised'
                                             'and semi supervised learning are provided.')

# paths to the training data
parser.add_argument('--data_path', type=str, help='path to the dataset, parent folder of the name of the dataset')
parser.add_argument('--dataset', type=str, help='name of the dataset')
parser.add_argument('--log_tag', type=str, help='experiment tag for the record')

# hyper parameters for training (both sup and semi sup):
parser.add_argument('--input_dim', type=int, help='dimension for the input image, e.g. 1 for CT, 3 for RGB, and more for 3D inputs', default=1)
parser.add_argument('--output_dim', type=int, help='dimension for the output, e.g. 1 for binary segmentation, 3 for 3 classes', default=1)
parser.add_argument('--full_train', type=bool, help='if true, train without validation, else, train with validation', default=True)
parser.add_argument('--iterations', type=int, help='number of iterations', default=10000)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--width', type=int, help='number of filters in the first conv block in encoder', default=32)
parser.add_argument('--depth', type=int, help='number of downsampling stages', default=4)
parser.add_argument('--batch', type=int, help='number of training batch size', default=5)
parser.add_argument('--temp', '--t', type=float, help='temperature scaling on output logits when applying sigmoid and softmax', default=2.0)
parser.add_argument('--l2', type=float, help='l2 normalisation', default=0.01)
parser.add_argument('--seed', type=int, help='random seed', default=1128)
parser.add_argument('--resolution', type=int, help='resolution for training images', default=512)
parser.add_argument('--saving_starting', type=int, help='number of iterations when it starts to save', default=2000)
parser.add_argument('--saving_frequency', type=int, help='number of interval of iterations when it starts to save', default=2000)

# hyper parameters for training (specific for semi sup)
parser.add_argument('--unlabelled', type=int, help='SSL, ratio between unlabelled and labelled data in one batch, if set up as 0, it will be supervised learning', default=2)
parser.add_argument('--detach', type=bool, help='SSL, true when we cut the gradients in consistency regularisation', default=True)
parser.add_argument('--mu', type=float, help='SSL, prior Gaussian mean', default=0.5) # mu
parser.add_argument('--std', type=float, help='SSL, prior Gaussian std', default=0.1) # sigma
parser.add_argument('--alpha', type=float, help='SSL, weight on the unsupervised learning part', default=1.0)
parser.add_argument('--warmup', type=float, help='SSL, ratio between the iterations of warming up and the whole training iterations', default=0.1)

# flags for data preprocessing and augmentation in data loader:
parser.add_argument('--norm', type=bool, help='true when using z score normalisation on images', default=True)
parser.add_argument('--contrast', type=bool, help='random contrast augmentation using histogram equalization', default=True)
# parser.add_argument('--lung_mask', type=bool, help='true when applying lung mask on the training data', default=False)

# flags for if we use fine-tuning on an trained model:
parser.add_argument('--resume', type=bool, help='resume training on an existing model', default=False)
parser.add_argument('--checkpoint_path', type=str, help='path to the checkpoint model')
