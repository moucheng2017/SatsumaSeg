# basic libs:
import argparse
from MainSemi import trainBPL
from MainSup import trainSup
# We use 0 or 1 for False or True as alternative for boolean operations in this argparse


def main():
    parser = argparse.ArgumentParser(description='Training for semi supervised segmentation with bayesian pseudo labels.')

    # paths to the training data
    parser.add_argument('--data', type=str, default='/home/moucheng/projects_data/hipct_covid/class1.0/', help='Data path')
    parser.add_argument('--format', type=str, default='np', help='np for numpy data; nii for nifti')
    parser.add_argument('--log_tag', type=str, default='2.5D', help='experiment tag for the record')

    # hyper parameters for training (both sup and semi sup):
    parser.add_argument('--input_dim', type=int, help='dimension for the input image, e.g. 1 for CT, 3 for RGB, and more for 3D inputs', default=1)
    parser.add_argument('--output_dim', type=int, help='dimension for the output, e.g. 1 for binary segmentation, 3 for 3 classes', default=1)
    parser.add_argument('--iterations', type=int, help='number of iterations', default=200000) # for covid, each epoch is roughly: 160 x 20 / batch,
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--width', type=int, help='number of filters in the first conv block in encoder', default=32)
    parser.add_argument('--depth', type=int, help='number of downsampling stages', default=4)
    parser.add_argument('--batch', type=int, help='number of training batch size', default=16)
    parser.add_argument('--temp', '--t', type=float, help='temperature scaling on output logits when applying sigmoid and softmax', default=2.0)
    parser.add_argument('--l2', type=float, help='l2 normalisation', default=1e-2)
    parser.add_argument('--seed', type=int, help='random seed', default=1128)
    parser.add_argument('--ema_saving_starting', type=int, help='number of iterations when it starts to save avg model', default=200)
    parser.add_argument('--patience', type=int, help='patience for validate accurate', default=2000) # about 10 epochs
    parser.add_argument('--validate_no', type=int, help='no of batch for validate because full validate is too time consuming', default=2)

    # hyper parameters for training (specific for semi sup)
    parser.add_argument('--unlabelled', type=int, help='SSL, ratio between unlabelled and labelled data in one batch, 0 for supervised learning', default=0)
    parser.add_argument('--mu', type=float, help='SSL, prior Gaussian mean', default=0.9)  # mu
    parser.add_argument('--alpha', type=float, help='SSL, weight on the unsupervised learning part', default=1.0)
    parser.add_argument('--beta', type=float, help='SSL, weight on the KL loss part', default=0.1)
    parser.add_argument('--warmup', type=float, help='SSL, ratio between the iterations of warming up and the whole training iterations', default=0.1)
    parser.add_argument('--warmup_start', type=int, help='SSL, when to start warm up the weight for the unsupervised learning part', default=160)

    # flags for data preprocessing and augmentation in data loader:
    parser.add_argument('--gaussian', type=int, help='1 when add random gaussian noise', default=1)
    parser.add_argument('--zoom', type=int, help='1 when use random zoom in augmentation', default=1)
    parser.add_argument('--cutout', type=int, help='1 when randomly cutout some patches', default=0)
    parser.add_argument('--contrast', type=int, help='1 when use random contrast using histogram equalization with random bins', default=1)
    parser.add_argument('--full_orthogonal', type=int, help='1 when each iteration has three orthogonal planes all together', default=1)
    parser.add_argument('--new_size_h', type=int, help='new size for the image height', default=160)
    parser.add_argument('--new_size_w', type=int, help='new size for the image width', default=160)

    # flags for if we use fine-tuning on an trained model:
    parser.add_argument('--resume', type=int, help='resume training on an existing model', default=0)
    parser.add_argument('--checkpoint_path', type=str, help='path to the checkpoint model')

    global args
    args = parser.parse_args()

    if args.unlabelled == 0:
        trainSup(args)
    else:
        trainBPL(args)


if __name__ == '__main__':
    main()


# disabled options:
# parser.add_argument('--lung_window', type=int, help='1 when we apply lung window on data', default=0)
# parser.add_argument('--sampling', type=int, help='weight for sampling the slices along each axis of 3d volume for training, '
#                                                  'highest weights at the edges and lowest at the middle', default=0)
# parser.add_argument('--norm', type=int, help='1 when normalise each case individually', default=1)
# parser.add_argument('--saving_frequency', type=int, help='number of interval of iterations when it starts to save', default=50)
# parser.add_argument('--cutout', type=int, help='1 when randomly cutout some patches', default=0)
# parser.add_argument('--detach', type=int, help='SSL, 1 when we cut the gradients in consistency regularisation or 0', default=0)
#     parser.add_argument('--sigma', type=float, help='SSL, prior Gaussian std', default=0.1)  # sigma