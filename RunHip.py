# basic libs:
import argparse
from MainSemiHip import trainBPL

# We use 0 or 1 for False or True as alternative for boolean operations in this argparse


def main():
    parser = argparse.ArgumentParser(description='Training for semi supervised segmentation with bayesian pseudo labels.')

    # paths to the training data
    parser.add_argument('--data', type=str, help='path to the dataset, parent folder of the name of the dataset')
    # parser.add_argument('--dataset', type=str, help='name of the dataset')
    parser.add_argument('--log_tag', type=str, help='experiment tag for the record')

    # hyper parameters for training (both sup and semi sup):
    parser.add_argument('--input_dim', type=int, help='dimension for the input image, e.g. 1 for CT, 3 for RGB, and more for 3D inputs', default=1)
    parser.add_argument('--output_dim', type=int, help='dimension for the output, e.g. 1 for binary segmentation, 3 for 3 classes', default=1)
    parser.add_argument('--iterations', type=int, help='number of iterations', default=10000)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--width', type=int, help='number of filters in the first conv block in encoder', default=16)
    parser.add_argument('--depth', type=int, help='number of downsampling stages', default=3)
    parser.add_argument('--batch', type=int, help='number of training batch size', default=4)
    parser.add_argument('--temp', '--t', type=float, help='temperature scaling on output logits when applying sigmoid and softmax', default=2.0)
    parser.add_argument('--l2', type=float, help='l2 normalisation', default=0.01)
    parser.add_argument('--seed', type=int, help='random seed', default=1128)
    parser.add_argument('--saving_starting', type=int, help='number of iterations when it starts to save', default=1000)
    parser.add_argument('--saving_frequency', type=int, help='number of interval of iterations when it starts to save', default=500)

    # hyper parameters for training (specific for semi sup)
    parser.add_argument('--unlabelled', type=int, help='SSL, ratio between unlabelled and labelled data in one batch, if set up as 0, it will be supervised learning', default=2)
    parser.add_argument('--detach', type=int, help='SSL, 1 when we cut the gradients in consistency regularisation or 0', default=0)
    parser.add_argument('--mu', type=float, help='SSL, prior Gaussian mean', default=0.9)  # mu
    parser.add_argument('--sigma', type=float, help='SSL, prior Gaussian std', default=0.1)  # sigma
    parser.add_argument('--alpha', type=float, help='SSL, weight on the unsupervised learning part', default=1.0)
    parser.add_argument('--beta', type=float, help='SSL, weight on the KL loss part', default=0.1)
    parser.add_argument('--warmup', type=float, help='SSL, ratio between the iterations of warming up and the whole training iterations', default=0.1)

    # flags for data preprocessing and augmentation in data loader:
    parser.add_argument('--norm', type=int, help='1 when normalise each case individually', default=1)
    parser.add_argument('--gaussian', type=int, help='1 when add random gaussian noise', default=1)
    parser.add_argument('--cutout', type=int, help='1 when randomly cutout some patches', default=1)
    parser.add_argument('--sampling', type=int, help='weight for sampling the slices along each axis of 3d volume for training, '
                                                     'highest weights at the edges and lowest at the middle', default=5)
    parser.add_argument('--zoom', type=int, help='1 when use random zoom in augmentation', default=1)
    parser.add_argument('--contrast', type=int, help='1 when use random contrast using histogram equalization with random bins', default=1)
    parser.add_argument('--lung_window', type=int, help='1 when we apply lung window on data', default=1)
    # parser.add_argument('--full_orthogonal', action='store_true', help='true when each iteration has three orthogonal planes all together')
    parser.add_argument('--full_orthogonal', type=int, help='1 when each iteration has three orthogonal planes all together', default=0)
    parser.add_argument('--new_size_h', type=int, help='new size for the image height', default=384)
    parser.add_argument('--new_size_w', type=int, help='new size for the image width', default=384)

    # flags for if we use fine-tuning on an trained model:
    parser.add_argument('--resume', type=int, help='resume training on an existing model', default=0)
    parser.add_argument('--checkpoint_path', type=str, help='path to the checkpoint model')

    global args
    args = parser.parse_args()

    trainBPL(args)


if __name__ == '__main__':
    main()