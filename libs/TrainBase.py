import torch
import torch.nn as nn
from Loss import SoftDiceLoss, kld_loss
from Metrics import segmentation_scores

# todo: I need to re write train_base for supervised part
# todo: Add a train_base for both supervised part and unsupervised part

# def calculate_sup_loss():


# def calculate_pl_loss():


def train_base(labelled_img,
               labelled_label,
               labelled_lung,
               device,
               model,
               model_base=True,
               t=2.0,
               apply_lung_mask=True,
               single_channel_label=False):
    '''
    Args:
        labelled_img:
        labelled_label:
        labelled_lung:
        device:
        model:
        model_base:
        t:
        apply_lung_mask:
        single_channel_label:

    Returns:
    '''

    train_imgs = labelled_img.to(device=device, dtype=torch.float32)
    labels = labelled_label.to(device=device, dtype=torch.float32)
    lung = labelled_lung.to(device=device, dtype=torch.float32)

    if single_channel_label is True:
        labels = labels.unsqueeze(1)
        lung = lung.unsqueeze(1)
        train_imgs = train_imgs.unsqueeze(1)

    outputs = model(train_imgs)

    if torch.sum(labels) > 10.0:
        # we ignore slices with way too little foregorund pixels as they are meaningless anyways
        if model_base is True: # this is plain u-net
            outputs = model(train_imgs)

        if outputs.size()[-1] == 1: # if the output is single channel that means we are using binary segmentation
            prob_outputs = torch.sigmoid(outputs / t)
        else:
            prob_outputs = torch.softmax(outputs / t, dim=1)
            prob_outputs = prob_outputs[:, -1, :, :].unsqueeze(1)

        if apply_lung_mask is True:
            lung_mask = (lung > 0.5) # float to bool
            prob_outputs_masked = torch.masked_select(prob_outputs, lung_mask)
            labels_masked = torch.masked_select(labels, lung_mask)
            loss = SoftDiceLoss()(prob_outputs_masked, labels_masked) + nn.BCELoss(reduction='mean')(prob_outputs_masked.squeeze() + 1e-10, labels_masked.squeeze() + 1e-10)
            class_outputs = (prob_outputs_masked > 0.95).float()
            train_mean_iu_ = segmentation_scores(labels_masked, class_outputs, 2)
            train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)
        else:
            loss = SoftDiceLoss()(prob_outputs, labels) + nn.BCELoss(reduction='mean')(prob_outputs.squeeze() + 1e-10, labels.squeeze() + 1e-10)
            class_outputs = (prob_outputs > 0.95).float()
            train_mean_iu_ = segmentation_scores(labels, class_outputs, 2)
            train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)

    else:
        train_mean_iu_ = 0.0
        loss = 0.0

    if model_base is True:
        return loss, train_mean_iu_
    else:
        return loss, train_mean_iu_, outputs


def train_ssl_base(labelled_img,
                   labelled_label,
                   labelled_lung,
                   unlabelled_img,
                   alpha,
                   warmup,
                   max_iterations,
                   device,
                   model,
                   model_base=True,
                   t=2.0,
                   apply_lung_mask=True,
                   single_channel_label=False):
    '''
    Args:
        labelled_img:
        labelled_label:
        labelled_lung:
        device:
        model:
        model_base:
        t:
        apply_lung_mask:
        single_channel_label:

    Returns:
    '''

    sup_loss, train_mean_iu_ = train_base(labelled_img, labelled_label, labelled_lung, device, model, model_base, t, apply_lung_mask, single_channel_label)
    # kl
    return sup_loss, train_mean_iu_