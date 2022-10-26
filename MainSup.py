from pathlib import Path
import torch
import timeit
import shutil
import math

from libs.Train import train_sup
from libs import Helpers


def trainSup(args):
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

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(args)

    # train labelled:
    train_labelled_data_loader = data_iterators.get('train_loader_l')
    iterator_train_labelled = iter(train_labelled_data_loader)

    # running loop:
    for step in range(args.iterations):
        # put model to training mode:
        model.train()

        # labelled data
        labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        if args.full_orthogonal == 1:
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

            loss = loss_d.get('supervised loss').get('loss') + loss_h.get('supervised loss').get('loss') + loss_w.get('supervised loss').get('loss')
            loss = loss.mean() / 3

            del labelled_dict

            if loss > 0.0:
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
                                                    loss_d.get('supervised loss').get('loss').mean().item(),
                                                    loss_h.get('supervised loss').get('loss').mean().item(),
                                                    loss_w.get('supervised loss').get('loss').mean().item(),))

                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #

                writer.add_scalars('loss metrics', {'train seg loss d': loss_d.get('supervised loss').get('loss').mean().item(),
                                                    'train seg loss h': loss_h.get('supervised loss').get('loss').mean().item(),
                                                    'train seg loss w': loss_w.get('supervised loss').get('loss').mean().item(),
                                                    }, step + 1)

        elif args.full_orthogonal == 0:
            loss_o = train_sup(labelled_img=labelled_dict["plaine"][0],
                               labelled_label=labelled_dict["plaine"][1],
                               model=model,
                               t=args.temp,
                               augmentation_cutout=args.cutout)

            loss = loss_o.get('supervised loss').get('loss').mean()

            del labelled_dict

            if loss > 0.0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for param_group in optimizer.param_groups:
                    # exponential decay
                    param_group["lr"] = args.lr * ((1 - float(step) / args.iterations) ** 0.99)

                print(
                    'Step [{}/{}], '
                    'lr: {:.4f},'
                    'Train loss: {:.4f}, '.format(step + 1,
                                                    args.iterations,
                                                    optimizer.param_groups[0]["lr"],
                                                    loss))

                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #

                writer.add_scalars('loss metrics', {'train loss': loss.get('supervised loss')}, step + 1)

        if step > args.saving_starting and step % args.saving_frequency == 0:
            save_model_name_full = saved_model_path + '/' + model_name + '_' + str(step) + '.pt'
            torch.save(model, save_model_name_full)

        elif step > args.iterations - 100 and step % 2 == 0:
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














