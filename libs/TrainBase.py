import torch
import torch.nn as nn
from Loss import SoftDiceLoss, kld_loss
from Metrics import segmentation_scores


def check_dim(input_tensor):
    if len(input_tensor.size()) < 4:
        return input_tensor.unsqueeze(1)
    else:
        return input_tensor


def check_inputs(img,
                 lbl,
                 lung):
    img = check_dim(img)
    lbl = check_dim(lbl)
    lung = check_dim(lung)
    return img, lbl, lung


def np2tensor_all(img,
                  lbl,
                  lung,
                  device='cuda'):
    img = img.to(device=device, dtype=torch.float32)
    lbl = lbl.to(device=device, dtype=torch.float32)
    lung = lung.to(device=device, dtype=torch.float32)
    img, lbl, lung = check_inputs(img, lbl, lung)
    return img, lbl, lung


def model_forward(model, img):
    return model(img)

def get_img_ssl(img_l, img_u):
    img = torch.cat((img_l, img_u), dim=0)
    b_l = img_l.size()[0]
    b_u = img_u.size()[0]
    del img_l
    del img_u
    return img, b_l, b_u

def calculate_sup_loss(outputs_dict,
                       lbl,
                       lung,
                       temp,
                       apply_lung_mask):

    if torch.sum(lbl) > 10.0:
        prob_output = outputs_dict.get('segmentation')

        if prob_output.size()[-1] == 1:  # if the output is single channel that means we are using binary segmentation
            prob_output = torch.sigmoid(prob_output / temp)
        else:
            prob_output = torch.softmax(prob_output / temp, dim=1)

        if apply_lung_mask is True:
            lung_mask = (lung > 0.5)  # float to bool
            prob_output = torch.masked_select(prob_output, lung_mask)
            lbl = torch.masked_select(lbl, lung_mask)

        loss = SoftDiceLoss()(prob_output, lbl) + nn.BCELoss(reduction='mean')(prob_output.squeeze() + 1e-10, lbl.squeeze() + 1e-10)
        class_outputs = (prob_output > 0.95).float()
        train_mean_iu_ = segmentation_scores(lbl, class_outputs, 2)
        train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)

    else:
        loss = 0.0
        train_mean_iu_ = 0.0

    return loss, train_mean_iu_



# def calculate_ssl_loss(outputs_dict,
#                        lbl,
#                        lung,
#                        temp,
#                        apply_lung_mask):





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