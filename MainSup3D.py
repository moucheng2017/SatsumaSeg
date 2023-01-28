from pathlib import Path
import torch
import timeit
import shutil
import math
import numpy as np

from libs.Train3D import train_sup
from libs import Helpers
from libs.Validate import validate

from arguments import get_args


def main(args):
    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)
    model_ema, _ = Helpers.network_intialisation(args)

    # resume training:
    if args.checkpoint.resume is True:
        model = torch.load(args.checkpoint.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    model_ema.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.train.optimizer.weight_decay)

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(args)

    # train labelled:
    train_labelled_data_loader = data_iterators.get('train_loader_l')
    iterator_train_labelled = iter(train_labelled_data_loader)

    # validate labelled:
    # val_labelled_data_loader = data_iterators.get('val_loader_l')

    # initialisation of best acc tracker
    best_val = 0.0

    # initialisation of counter for ema avg:
    ema_count = 0

    # initialisation of growth of val acc tracker
    best_val_count = 1

    # return {'loss_sup': loss_sup,
    #         'loss_unsup': loss_unsup,
    #         'train iou': train_mean_iu_}

    # running loop:
    for step in range(args.train.iterations):

        # initialisation of validating acc:
        validate_acc = 0.0

        # put model to training mode:
        model.train()

        # labelled data
        labelled_data = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        # ratio of unlabelled data:
        # mask_foreground = np.zeros_like(labelled_data.get('lbl'))
        # mask_foreground[labelled_data.get('lbl') > 0] = 1
        # b, d, h, w = np.shape(mask_foreground)
        # foreground_ratio = sum(sum(sum(sum(mask_foreground)))) / (b*d*h*w)
        # print(foreground_ratio)

        loss_ = train_sup(labelled_img=labelled_data.get('img'),
                          labelled_label=labelled_data.get('lbl'),
                          model=model,
                          t=args.train.temp)

        loss = loss_.get('supervised losses').get('loss_sup').mean()
        # loss_unsup = loss_.get('supervised losses').get('loss_unsup').mean()

        # if args.use_pseudo == 1:
        #     if step < 0.3*args.iterations:
        #         loss = loss_sup
        #     elif step < 0.6*args.iterations:
        #         ratio = (step - 0.3*args.iterations) / (0.6*args.iterations - 0.3*args.iterations)
        #         loss = loss_sup + loss_unsup*(1 - foreground_ratio)*ratio
        #     else:
        #         loss = loss_sup + (1 - foreground_ratio)*loss_unsup
        # else:
        #     loss = loss_sup

        train_iou = loss_.get('supervised losses').get('train iou')

        del labelled_data

        if loss > 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                # exponential decay
                param_group["lr"] = args.train.lr * ((1 - float(step) / args.train.iterations) ** 0.99)

            # validate_acc = validate(validate_loader=val_labelled_data_loader,
            #                         model=model,
            #                         no_validate=args.validate_no,
            #                         full_orthogonal=args.full_orthogonal)

            print(
                'Step [{}/{}], '
                'lr: {:.4f},'
                'train iou: {:.4f},'
                'Train loss: {:.4f}, '.format(step + 1,
                                              args.train.iterations,
                                              optimizer.param_groups[0]["lr"],
                                              train_iou,
                                              loss))

            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

            writer.add_scalars('loss metrics', {'train loss': loss.item()}, step + 1)
            writer.add_scalars('ious', {'train iu': train_iou,
                                        'val iu': validate_acc}, step + 1)

        if step > args.train.save_ema_start:
            ema_count += 1
            for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.add_(param.data)

        if validate_acc > best_val:
            save_model_name_full = saved_model_path + '/' + model_name + '_best_val.pt'
            torch.save(model, save_model_name_full)
            best_val = max(best_val, validate_acc)
        else:
            best_val_count += 1
            best_val = best_val
            if best_val_count > args.train.patience and best_val > 0.95:
                for ema_param in model_ema.parameters():
                    ema_param = ema_param / ema_count
                save_model_name_full = saved_model_path + '/' + model_name + '_ema.pt'
                torch.save(model_ema, save_model_name_full)
                break
            else:
                save_model_name_full = saved_model_path + '/' + model_name + '_ema.pt'
                torch.save(model_ema, save_model_name_full)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    # save_path = saved_model_path + '/results'
    # Path(save_path).mkdir(parents=True, exist_ok=True)
    # print('\nTraining finished and model saved\n')
    #
    # # zip all models:
    # shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    # shutil.rmtree(saved_model_path)


if __name__ == "__main__":
    args = get_args()
    main(args=args)

    # completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    #
    # os.rename(args.log_dir, completed_log_dir)
    # print(f'Log file has been saved to {completed_log_dir}')









