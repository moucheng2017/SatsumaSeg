# basic libs:
import os
import math
import errno
import timeit
import shutil
from pathlib import Path

# deterministic training:
import random
import numpy as np
import torch.backends.cudnn as cudnn

# deep learning:
import torch
import torch.nn as nn
from Models2D import UnetBPL
from libs.DataloaderOrthogonal import getData

# metrics
from Metrics import segmentation_scores

# log
from tensorboardX import SummaryWriter

# evaluation
from libs.Utils import evaluate, sigmoid_rampup


# training options control panel:
from TrainArguments import parser


def main():
    global args
    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for j in range(1, args.repeat + 1):

        repeat_str = str(j)

        Exp = UnetBPL(in_ch=args.input_dim,
                      width=args.width,
                      depth=args.depth,
                      out_ch=args.output_dim,
                      norm='in',
                      ratio=4,
                      detach=args.detach)

        Exp_name = 'BayesianPLSeg2DOrthogonal'

        Exp_name = Exp_name + \
                   '_e' + str(repeat_str) + \
                   '_l' + str(args.lr) + \
                   '_m' + str(args.mu) + \
                   '_v' + str(args.sigma) + \
                   '_b' + str(args.batch) + \
                   '_u' + str(args.unlabelled) + \
                   '_w' + str(args.width) + \
                   '_d' + str(args.depth) + \
                   '_i' + str(args.iterations) + \
                   '_r' + str(args.l2) + \
                   '_a' + str(args.alpha) + \
                   '_w' + str(args.warmup) + \
                   '_det' + str(args.detach)

        data = getData(data_directory=args.data_path,
                       dataset_name=args.dataset,
                       train_batchsize=args.batch,
                       norm=args.norm,
                       contrast_aug=args.contrast,
                       lung_window=True,
                       resolution=args.resolution,
                       train_full=True,
                       unlabelled=True)

        trainSingleModel(model=Exp,
                         model_name=Exp_name,
                         num_iterations=args.iterations,
                         learning_rate=args.lr,
                         dataset_name=args.dataset,
                         trainloader_with_labels=data.get('train_loader_l'),
                         trainloader_without_labels=data.get('train_loader_u'),
                         log_tag=args.log_tag,
                         l2=args.l2,
                         alpha=args.alpha,
                         warmup=args.warmup,
                         mean_prior=args.mu,
                         std_prior=args.std
                         )


def trainSingleModel(model,
                     model_name,
                     num_iterations,
                     learning_rate,
                     dataset_name,
                     trainloader_with_labels,
                     trainloader_without_labels,
                     log_tag,
                     l2=0.01,
                     alpha=1.0,
                     warmup=0.1,
                     mean_prior=0.5,
                     std_prior=0.1):

    device = torch.device('cuda')
    save_model_name = model_name
    saved_information_path = '../Results/' + dataset_name + '/' + log_tag
    Path(saved_information_path).mkdir(parents=True, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    Path(saved_log_path).mkdir(parents=True, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    Path(saved_model_path).mkdir(parents=True, exist_ok=True)

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)

    start = timeit.default_timer()

    iterator_train_labelled = iter(trainloader_with_labels)
    iterator_train_unlabelled = iter(trainloader_without_labels)

    train_mean_iu_d_tracker = 0.0
    train_mean_iu_h_tracker = 0.0
    train_mean_iu_w_tracker = 0.0

    for step in range(num_iterations):

        model.train()

        train_iou = []
        train_sup_loss = []
        train_unsup_loss = []
        train_kl_loss = []

        if step < int(warmup * num_iterations):
            scale = sigmoid_rampup(step, int(warmup * num_iterations), 1.0)
            alpha_current = alpha * scale
        else:
            alpha_current = alpha

        try:
            labelled_img, labelled_label, labelled_lung, labelled_name = next(iterator_train_labelled)
            unlabelled_img, unlabelled_lung, unlabelled_name = next(iterator_train_unlabelled)
        except StopIteration:
            iterator_train_labelled = iter(trainloader_with_labels)
            labelled_img, labelled_label, labelled_lung, labelled_name = next(iterator_train_labelled)

            iterator_train_unlabelled = iter(trainloader_without_labels)
            unlabelled_img, unlabelled_lung, unlabelled_name = next(iterator_train_unlabelled)

        train_imgs_l = labelled_img.to(device=device, dtype=torch.float32)
        b_l = train_imgs_l.size()[0]

        train_imgs_u = unlabelled_img.to(device=device, dtype=torch.float32)
        b_u = train_imgs_u.size()[0]

        # print(train_imgs_u.size())
        # print(train_imgs_l.size())

        train_imgs = torch.cat((train_imgs_l, train_imgs_u), dim=0)

        labels = labelled_label.to(device=device, dtype=torch.float32)

        labelled_lung = labelled_lung.to(device=device, dtype=torch.float32)
        unlabelled_lung = unlabelled_lung.to(device=device, dtype=torch.float32)

        if torch.sum(labels) > 10.0:
            # forward pass of segmentation model, outputs from the last the second last layers
            outputs, threshold_input = model(train_imgs, [dilation, dilation, dilation, dilation], [dilation, dilation, dilation, dilation])
            # forward pass of threshold model
            mu, logvar = model_t(threshold_input)
            # split outputs between labelled and unlabelled for segmentation
            outputs_l, outputs_u = torch.split(outputs, [b_l, b_u], dim=0)
            # split outputs between labelled and unlabelled for means of threshold
            mu_l, mu_u = torch.split(mu, [b_l, b_u], dim=0)
            # split outputs between labelled and unlabelled for logvars of threshold (DO NOT ADD RELU AFTER LOGVAR!!! It is supposed to be negative!!!)
            logvar_l, logvar_u = torch.split(logvar, [b_l, b_u], dim=0)

            # reparametrization of threshold for labelled data
            std_l = torch.exp(0.5 * logvar_l)
            eps_l = torch.rand_like(std_l)
            threshold_learnt_l = mu_l + eps_l * std_l

            # reparametrization of threshold for unlabelled data
            std_u = torch.exp(0.5 * logvar_u)
            eps_u = torch.rand_like(std_u)
            threshold_learnt_u = mu_u + eps_u * std_u

            # pseudo labelling for labelled data
            prob_outputs_l = torch.sigmoid(outputs_l)
            # supervised learning on labelled data
            lung_mask_labelled = (labelled_lung > 0.5)
            prob_outputs_l_masked = torch.masked_select(prob_outputs_l, lung_mask_labelled)
            labels_masked = torch.masked_select(labels, lung_mask_labelled)

            if torch.sum(prob_outputs_l_masked) > 10.0:
                loss_s = SoftDiceLoss()(prob_outputs_l_masked, labels_masked) + nn.BCELoss(reduction='mean')(prob_outputs_l_masked.squeeze()+1e-10, labels_masked.squeeze()+1e-10)
            else:
                loss_s = 0.0

            if loss_s != 0.0:
                train_sup_loss.append(loss_s.item())
            else:
                train_sup_loss.append(0.0)

            # segmentation result on training labelled data
            class_outputs = (prob_outputs_l_masked > 0.5).float()
            # train iou
            train_mean_iu_ = segmentation_scores(labels_masked, class_outputs, class_no)
            train_iou.append(train_mean_iu_)
            # validate iou
            validate_iou, validate_h_dist = evaluate(validateloader, model, device, model_name, class_no, dilation)
            # E-step: pseudo labelling for unlabelled data
            prob_outputs_u = torch.sigmoid(outputs_u)
            pseudo_label_u = (prob_outputs_u.detach() > threshold_learnt_u).float()
            # applying lung mask
            lung_mask_unlabelled = (unlabelled_lung > 0.5)
            prob_outputs_u_masked = torch.masked_select(prob_outputs_u, lung_mask_unlabelled)
            pseudo_label_u_masked = torch.masked_select(pseudo_label_u, lung_mask_unlabelled)

            # loss for unsupervised data with their pseudo labels
            # foreground_in_pseudo_labels_l = [torch.sum(pseudo_label_l_masked[i, :, :, :, :].detach()) for i in range(pseudo_label_l_masked.size()[0])]
            # foreground_in_pseudo_labels_u = [torch.sum(pseudo_label_u_masked[i, :, :, :, :].detach()) for i in range(pseudo_label_u_masked.size()[0])]
            loss_u = 0.0
            # for i, foreground_index in enumerate(foreground_in_pseudo_labels_u):
            if 10.0 < torch.sum(pseudo_label_u_masked) < torch.numel(pseudo_label_u_masked):
                loss_u += SoftDiceLoss()(prob_outputs_u_masked, pseudo_label_u_masked) + nn.BCELoss(reduction='mean')(prob_outputs_u_masked.squeeze() + 1e-10, pseudo_label_u_masked.squeeze() + 1e-10)

                # loss_u += SoftDiceLoss()(prob_outputs_u_masked, pseudo_label_u_masked)
            # # a regularisation for supervised data with their pseudo labels
            # if 10.0 < torch.sum(pseudo_label_l_masked) < torch.numel(pseudo_label_l_masked):
            #     # loss_u += 0.1 * SoftDiceLoss()(prob_outputs_l_masked, pseudo_label_l_masked) + 0.1 * nn.BCELoss(reduction='mean')(prob_outputs_l_masked.squeeze() + 1e-10, pseudo_label_l_masked.squeeze() + 1e-10)
            #     loss_u += SoftDiceLoss()(prob_outputs_l_masked, pseudo_label_l_masked) + nn.BCELoss(reduction='mean')(prob_outputs_l_masked.squeeze() + 1e-10, pseudo_label_l_masked.squeeze() + 1e-10)
            # weighting the pseudo label losses

            loss_u = loss_u*alpha_current
            if loss_u != 0.0:
                train_unsup_loss.append(loss_u.item())
            else:
                train_unsup_loss.append(0.0)

            # final loss
            loss = loss_u + loss_s

            # M-step: updating the segmentation model
            if loss != 0.0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # K-L loss for threshold model:
            kld_loss = math.log(std_prior) - logvar + 0.5 * (logvar.exp() + (mu - mean_prior).pow(2)) / std_prior**2 - 0.5
            kld_loss = kld_loss.mean() * alpha_current
            train_kl_loss.append(kld_loss.item())
            # update the threshold model
            optimizer_t.zero_grad()
            kld_loss.backward()
            optimizer_t.step()

            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * ((1 - float(step) / num_steps) ** 0.99)

            print(
                'Step [{}/{}], '
                'lr: {:.4f},'
                'threshold_l: {:.4f}, '
                'threshold_u: {:.4f}, '
                'Train sup loss: {:.4f}, '
                'Train unsup loss: {:.4f}, '
                'Train kl loss: {:.4f}, '
                'Train iou: {:.4f}, '
                'val iou:{:.4f}, '.format(step + 1, num_steps,
                                          optimizer.param_groups[0]["lr"],
                                          float(threshold_learnt_l.cpu().detach().mean()),
                                          float(threshold_learnt_u.cpu().detach().mean()),
                                          np.nanmean(train_sup_loss),
                                          np.nanmean(train_unsup_loss),
                                          np.nanmean(train_kl_loss),
                                          np.nanmean(train_iou),
                                          np.nanmean(validate_iou)))

            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

            writer.add_scalars('acc metrics', {'train iou': np.nanmean(train_iou),
                                               'val iou': np.nanmean(validate_iou)}, step + 1)

            writer.add_scalars('loss values', {'sup loss': np.nanmean(train_sup_loss),
                                               'unsup loss': np.nanmean(train_unsup_loss)}, step + 1)

            writer.add_scalars('hyperparameter values', {'alpha current': alpha_current}, step + 1)

            writer.add_scalars('hyperparameter values', {'threshold current labelled': float(threshold_learnt_l.cpu().detach().mean()),
                                                         'threshold current unlabelled': float(threshold_learnt_u.cpu().detach().mean())}, step + 1)

        if step > 2000 and step % 50 == 0 or step > num_steps - 50:
            # save checker points
            save_model_name_full = saved_model_path + '/' + save_model_name + '_' + str(step) + '.pt'
            path_model = save_model_name_full
            torch.save(model, path_model)

    # save_model_name_full = saved_model_path + '/' + save_model_name + '.pt'
    # path_model = save_model_name_full
    # torch.save(model, path_model)

    stop = timeit.default_timer()
    training_time = stop - start
    print('Training Time: ', training_time)

    save_path = saved_model_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # iou_mean, iou_std = test_all_models(saved_model_path, testdata_path, save_path, size, class_no, False, False)
    # print('Test IoU: ' + str(iou_mean) + '\n')
    # print('Test IoU std: ' + str(iou_std) + '\n')

    print('\nTraining finished and model saved\n')

    # zip all models:
    shutil.make_archive(saved_model_path, 'zip', saved_model_path)
    shutil.rmtree(saved_model_path)

    return model
