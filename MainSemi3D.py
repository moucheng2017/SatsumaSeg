from pathlib import Path
import torch
import timeit
import shutil

from libs.Train3D import train_semi
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
    # if args.resume == 1:
    #     model = torch.load(args.checkpoint_path)

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
    val_labelled_data_loader = data_iterators.get('val_loader_l')

    # train unlabelled:
    train_unlabelled_data_loader = data_iterators.get('train_loader_u')
    iterator_train_unlabelled = iter(train_unlabelled_data_loader)

    # initialisation of best acc tracker
    best_val = 0.0

    # initialisation of counter for ema avg:
    ema_count = 0

    # initialisation of growth of val acc tracker
    best_val_count = 1

    # running loop:
    for step in range(args.train.iterations):

        # initialisation of validating acc:
        validate_acc = 0.0

        # ramp up alpha and beta:
        current_alpha = Helpers.ramp_up(args.train.alpha, args.train.warmup, step, args.train.iterations, args.train.warmup_start)
        # current_beta = Helpers.ramp_up(args.beta, args.warmup, step, args.train.iterations, args.warmup_start)

        # put model to training mode:
        model.train()
        model_ema.train()

        # labelled data
        labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        # unlabelled data:
        unlabelled_dict = Helpers.get_data_dict(train_unlabelled_data_loader, iterator_train_unlabelled)

        loss_ = train_semi(labelled_img=labelled_dict.get('img'),
                           labelled_label=labelled_dict.get('lbl'),
                           unlabelled_img=unlabelled_dict.get('img'),
                           model=model,
                           t=args.train.temp,
                           prior_mu=args.train.mu)

        sup_loss = loss_.get('supervised losses').get('loss_sup').mean()
        train_iou = loss_.get('supervised losses').get('train iou')
        # print(sup_loss)

        pseudo_loss = current_alpha*loss_.get('pseudo losses').get('loss_unsup').mean()

        # kl_loss = current_alpha*current_beta*loss_o.get('kl loss').get('loss').mean()

        loss = sup_loss + 0.1*pseudo_loss

        # learnt_threshold = loss_.get('kl loss').get('threshold unlabelled').mean()
        # print(learnt_threshold)

        del labelled_dict

        if sup_loss > 0.0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.train.lr * ((1 - float(step) / args.train.iterations) ** 0.99)

            validate_acc = validate(validate_loader=val_labelled_data_loader,
                                    model=model,
                                    no_validate=args.validate_no,
                                    full_orthogonal=args.full_orthogonal)

            # 'kl loss: {:.4f}, '
            # 'Threshold: {:.4f}'.

            print(
                'Step [{}/{}], '
                'lr: {:.4f},'
                'train iou: {:.4f},'
                'val iou: {:.4f},'
                'loss: {:.4f}, '
                'pseudo loss: {:.4f} '.format(step + 1,
                                           args.train.iterations,
                                           optimizer.param_groups[0]["lr"],
                                           train_iou,
                                           validate_acc,
                                           sup_loss.item(),
                                           pseudo_loss.item())
            )

            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

            # writer.add_scalars('loss metrics', {'train seg loss': loss_o.get('supervised loss').get('loss'),
            #                                     'train pseudo loss': pseudo_loss,
            #                                     'learnt threshold': learnt_threshold,
            #                                     'train kl loss': kl_loss}, step + 1)
            writer.add_scalars('ious', {'train iu': train_iou,
                                        'val iu': validate_acc}, step + 1)

            writer.add_scalars('loss metrics', {'train seg loss': sup_loss,
                                                'train pseudo loss': pseudo_loss}, step + 1)

        if step > args.ema_saving_starting:
            ema_count += 1
            if (step - args.ema_saving_starting) == 1:
                for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                    ema_param.data = param.data
            else:
                for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                    ema_param.data.add_(param.data)

        if validate_acc > best_val:
            save_model_name_full = saved_model_path + '/' + model_name + '_best_val.pt'
            torch.save(model, save_model_name_full)
            best_val = max(best_val, validate_acc)
        else:
            best_val_count += 1
            best_val = best_val
            if best_val_count > args.patience and best_val > 0.95:
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

    # # zip all models:
    # shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    # shutil.rmtree(saved_model_path)


if __name__ == "__main__":
    args = get_args()
    main(args=args)












