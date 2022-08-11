import torch
import torch.nn as nn
from Loss import SoftDiceLoss, kld_loss
from Metrics import segmentation_scores


def check_dim(input_tensor):
    if len(input_tensor.size()) < 4:
        return input_tensor.unsqueeze(1)
    else:
        return input_tensor


def check_inputs(img_l,
                 lbl,
                 lung,
                 img_u=None):
    img_l = check_dim(img_l)
    lbl = check_dim(lbl)
    lung = check_dim(lung)
    if img_u is not None:
        img_u = check_dim(img_u)
        return {'img_l': img_l,
                'img_u': img_u,
                'lung': lung,
                'lbl': lbl}
    else:
        return {'img_l': img_l,
                'img_u': None,
                'lbl': lbl,
                'lung': lung}


def np2tensor_all(img_l,
                  lbl,
                  lung,
                  img_u=None,
                  device='cuda'):
    '''
    Put all numpy files into tensor
    Args:
        img:
        lbl:
        lung:
        device:
    Returns:
    '''
    img_l = img_l.to(device=device, dtype=torch.float32)
    lbl = lbl.to(device=device, dtype=torch.float32)
    lung = lung.to(device=device, dtype=torch.float32)
    if img_u is None:
        inputs = check_inputs(img_l, lbl, lung)
    else:
        inputs = check_inputs(img_l, lbl, lung, img_u)
    return inputs


def get_img(inputs):
    img_l = inputs.get('img_l')
    img_u = inputs.get('img_u')
    if img_u is not None:
        img = torch.cat((img_l, img_u), dim=0)
        b_l = img_l.size()[0]
        b_u = img_u.size()[0]
        del img_l
        del img_u
        return {'train img': img,
                'batch labelled': b_l,
                'batch unlabelled': b_u}
    else:
        return {'train img': img_l}


def model_forward(model, img):
    return model(img)


def calculate_sup_loss(outputs_dict,
                       lbl,
                       lung,
                       temp,
                       apply_lung_mask):
    '''
    Supervised loss part.
    Args:
        outputs_dict:
        lbl:
        lung:
        temp:
        apply_lung_mask:

    Returns:

    '''
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

    return {'seg loss': loss,
            'train iou': train_mean_iu_}


def calculate_ssl_loss(outputs_dict,
                       b_l,
                       b_u,
                       prior_u,
                       prior_var):
    '''
    Semi supervised loss part.
    Args:
        outputs_dict:
        b_l:
        b_u:
        prior_u:
        prior_var:

    Returns:

    '''
    posterior_mu = outputs_dict.get('mu')
    posterior_logvar = outputs_dict.get('logvar')

    threshold_mu = outputs_dict.get('threshold_mu')
    threshold_logvar = outputs_dict.get('threshold_logvar')

    loss = kld_loss(posterior_mu, posterior_logvar, prior_u, prior_var)

    std = torch.exp(0.5 * threshold_logvar)
    eps = torch.rand_like(std)

    threshold_learnt = threshold_mu + eps * std

    threshold_learnt_l, threshold_learnt_u = torch.split(threshold_learnt, [b_l, b_u], dim=0)

    return {'kl loss': loss,
            'threshold labelled': threshold_learnt_l,
            'threshold unlabelled': threshold_learnt_u}


def train_base(labelled_img,
               labelled_label,
               labelled_lung,
               model,
               unlabelled_img=None,
               t=2.0,
               prior_mu=0.4,
               prior_logsigma=0.1,
               apply_lung_mask=True):
    '''

    Args:
        labelled_img:
        labelled_label:
        labelled_lung:
        model:
        unlabelled_img:
        t:
        prior_mu:
        prior_logsigma:
        apply_lung_mask:

    Returns:

    '''
    # convert data from numpy to tensor:
    inputs = np2tensor_all(img_l=labelled_img, img_u=unlabelled_img, lbl=labelled_label, lung=labelled_lung)

    # concatenate labelled and unlabelled for ssl otherwise just use labelled img
    train_img = get_img(inputs)

    # forward pass:
    outputs_dict = model_forward(model, train_img.get('train img'))

    # supervised loss:
    sup_loss = calculate_sup_loss(outputs_dict=outputs_dict,
                                  lbl=inputs.get('lbl'),
                                  lung=inputs.get('lung'),
                                  temp=t,
                                  apply_lung_mask=apply_lung_mask)

    if unlabelled_img is None:
        return {'supervised loss': sup_loss}

    else:
        # pseudo label loss:
        pseudo_loss = calculate_ssl_loss(outputs_dict=outputs_dict,
                                         b_l=train_img.get('batch labelled'),
                                         b_u=train_img.get('batch unlabelled'),
                                         prior_u=prior_mu,
                                         prior_var=prior_logsigma)
        return {'supervised losses': sup_loss,
                'pseudo losses': pseudo_loss}


