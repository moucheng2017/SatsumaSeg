# basic libs:
import argparse
from pathlib import Path
import torch
import timeit
import shutil
import math

from libs.Train import train_sup
from libs import Helpers


def trainSup():
    parser = argparse.ArgumentParser(description='Training options for supervised segmentation.')

    # paths to the training data
    parser.add_argument('--data_path', type=str, help='path to the dataset, parent folder of the name of the dataset')
    parser.add_argument('--dataset', type=str, help='name of the dataset')
    parser.add_argument('--log_tag', type=str, help='experiment tag for the record')

    # hyper parameters for training (both sup and semi sup):
    parser.add_argument('--input_dim', type=int, help='dimension for the input image, e.g. 1 for CT, 3 for RGB, and more for 3D inputs', default=1)
    parser.add_argument('--output_dim', type=int, help='dimension for the output, e.g. 1 for binary segmentation, 3 for 3 classes', default=1)
    # parser.add_argument('--full_train', type=bool, help='if true, train without validation, else, train with validation', default=True) # we don't really have enough data to be split into train and val
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

    # flags for data preprocessing and augmentation in data loader:
    parser.add_argument('--norm', type=bool, help='true when normalise each case individually', default=True)
    parser.add_argument('--gaussian', type=bool, help='true when add random gaussian noise', default=True)
    parser.add_argument('--cutout', type=bool, help='true when randomly cutout some patches', default=True)
    parser.add_argument('--sampling', type=int, help='weight for sampling the slices along each axis of 3d volume for training, '
                                                     'highest weights at the edges and lowest at the middle', default=5)
    parser.add_argument('--zoom', type=bool, help='true when use random zoom in augmentation', default=True)
    parser.add_argument('--contrast', type=bool, help='true when use random contrast using histogram equalization with random bins', default=True)

    # flags for if we use fine-tuning on an trained model:
    parser.add_argument('--resume', type=bool, help='resume training on an existing model', default=False)
    parser.add_argument('--checkpoint_path', type=str, help='path to the checkpoint model')

    global args
    args = parser.parse_args()

    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)

    # resume training:
    if args.resume is True:
        model = torch.load(args.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2)

    # data loaders:
    data_loaders = Helpers.get_data_simple_wrapper(args)

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(data_loaders, args)

    # train labelled:
    train_labelled_data_loader = data_iterators.get('train_loader_l')
    iterator_train_labelled = iter(train_labelled_data_loader)

    # running loop:
    for step in range(args.iterations):
        # put model to training mode:
        model.train()

        # labelled data
        labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        loss_d = train_sup(labelled_img=labelled_dict["plaine_d"][0],
                           labelled_label=labelled_dict["plaine_d"][1],
                           model=model,
                           t=args.temp,
                           augmentation_cutout=args.cutout)

        loss_h = train_sup(labelled_img=labelled_dict["plaine_h"][0],
                           labelled_label=labelled_dict["plaine_h"][1],
                           model=model,
                           t=args.temp,
                           augmentation_cutout=args.cutout)

        loss_w = train_sup(labelled_img=labelled_dict["plaine_w"][0],
                           labelled_label=labelled_dict["plaine_w"][1],
                           model=model,
                           t=args.temp,
                           augmentation_cutout=args.cutout)

        loss = loss_d.get('supervised loss') + loss_h.get('supervised loss') + loss_w.get('supervised loss')
        loss = loss / 3

        del labelled_dict

        if loss != 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            # exponential decay
            param_group["lr"] = args.lr * ((1 - float(step) / args.iterations) ** 0.99)

        print(
            'Step [{}/{}], '
            'lr: {:.4f},'
            'Train loss d: {:.4f}, '
            'Train loss h: {:.4f}, '
            'Train loss w: {:.4f}, '.format(step + 1,
                                            args.iterations,
                                            optimizer.param_groups[0]["lr"],
                                            loss_d.get('supervised loss'),
                                            loss_h.get('supervised loss'),
                                            loss_w.get('supervised loss')))

        # # # ================================================================== #
        # # #                        TensorboardX Logging                        #
        # # # # ================================================================ #

        writer.add_scalars('loss metrics', {'train loss d': loss_d.get('supervised loss'),
                                            'train loss h': loss_h.get('supervised loss'),
                                            'train loss w': loss_w.get('supervised loss')}, step + 1)

        if step > args.saving_starting and step % args.saving_frequency == 0:
            save_model_name_full = saved_model_path + '/' + model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

        elif step > args.iterations - 50 and step % 2 == 0:
            save_model_name_full = saved_model_path + '/' + model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

    save_model_name_full = saved_model_path + '/' + model_name + '_' + str(args.iterations) + '.pt'
    torch.save(model, save_model_name_full)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    save_path = saved_model_path + '/results'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)














