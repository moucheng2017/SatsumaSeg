import torch
import torch.nn as nn
from Loss import SoftDiceLoss, kld_loss
from Metrics import segmentation_scores

from libs.Augmentations import RandomCutOut
from libs.LabelEncoding import multi_class_label_processing


def check_dim(input_tensor):
    '''
    Args:
        input_tensor:
    Returns:
    '''
    if len(input_tensor.size()) < 4:
        return input_tensor.unsqueeze(1)
    else:
        return input_tensor


# def check_inputs(img_l,
#                  lbl,
#                  *img_u):
#     '''
#     Args:
#         img_l:
#         lbl:
#         lung:
#         img_u:
#     Returns:
#     '''
#     img_l = check_dim(img_l)
#     lbl = check_dim(lbl)
#
#     if img_u:
#         img_u = check_dim(img_u[0])
#         return {'img_l': img_l,
#                 'img_u': img_u,
#                 'lbl': lbl}
#     else:
#         return {'img_l': img_l,
#                 'lbl': lbl}


def check_inputs(**kwargs):
    outputs = {}
    for key, val in kwargs.items():
        # check the dimension for each input
        outputs[key] = check_dim(val)
    return outputs


# def np2tensor_all(**kwargs):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     for key, val in kwargs.items():
#         val
#     img_l = img_l.to(device=device, dtype=torch.float32)
#     lbl = lbl.to(device=device, dtype=torch.float32)
#     # lung = lung.to(device=device, dtype=torch.float32)
#     if img_u:
#         inputs = check_inputs(img_l, lbl)
#     else:
#         inputs = check_inputs(img_l, lbl, img_u)
#     return inputs


# def np2tensor_all(img_l,
#                   lbl,
#                   *img_u):
#     '''
#
#     Args:
#         img_l:
#         lbl:
#         lung:
#         img_u:
#         device:
#
#     Returns:
#
#     '''
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     img_l = img_l.to(device=device, dtype=torch.float32)
#     lbl = lbl.to(device=device, dtype=torch.float32)
#     # lung = lung.to(device=device, dtype=torch.float32)
#     if img_u:
#         inputs = check_inputs(img_l, lbl)
#     else:
#         inputs = check_inputs(img_l, lbl, img_u)
#     return inputs


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
                       temp,
                       b_l,
                       b_u,
                       cutout_aug,
                       foreground_threshold=50):
    '''
    Args:
        outputs_dict:
        lbl:
        lung:
        temp:
        b_l:
        b_u:
        cutout_aug:
        apply_lung_mask:
        foreground_threshold:
    Returns:
    '''
    if torch.sum(lbl) > foreground_threshold: # check whether there are enough foreground pixels
        prob_output = outputs_dict.get('segmentation') # get segmentation map
        if b_u > 0: # check if unlabelled data is included
            prob_output, _ = torch.split(prob_output, [b_l, b_u], dim=0) # if both labelled and unlabelled, split the data and use the labelled

        prob_output = torch.sigmoid(prob_output / temp) # apply element-wise sigmoid

        # if apply_lung_mask is True: # apply lung mask
        #     lung_mask = (lung > 0.5)  # float to bool
        #     prob_output = torch.masked_select(prob_output, lung_mask)
        #     lbl = torch.masked_select(lbl, lung_mask)

        if cutout_aug is True: # apply cutout augmentation
            cutout = RandomCutOut()
            prob_output, lbl = cutout.cutout_seg(prob_output, lbl)

        # channel-wise loss (this is for multi-channel sigmoid function as well):
        if len(prob_output.size()) == 3:
            # this is binary segmentation
            loss = SoftDiceLoss()(prob_output, lbl) + nn.BCELoss(reduction='mean')(prob_output.squeeze() + 1e-10, lbl.squeeze() + 1e-10)
            class_outputs = (prob_output > 0.95).float()
            train_mean_iu_ = segmentation_scores(lbl, class_outputs, 2)
            train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)
            return {'seg loss': loss,
                    'train iou': train_mean_iu_}

        elif len(prob_output.size()) == 4:
            if prob_output.size()[1] == 1:
                # this is also binary segmentation
                loss = SoftDiceLoss()(prob_output, lbl) + nn.BCELoss(reduction='mean')(prob_output.squeeze() + 1e-10, lbl.squeeze() + 1e-10)
                class_outputs = (prob_output > 0.95).float()
                train_mean_iu_ = segmentation_scores(lbl, class_outputs, 2)
                train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)
                return {'seg loss': loss,
                        'train iou': train_mean_iu_}

            else:
                # this is multi class segmentation
                lbl = multi_class_label_processing(lbl, prob_output.size()[1]) # convert single channel multi integer class label to multi channel binary label
                loss = 0
                train_mean_iu_ = 0
                effective_classes = 0
                for i in range(prob_output.size()[1]): # multiple
                    if torch.sum(lbl[:, i, :, :]) > 1.0:
                        # If the channel is not empty, we learn it otherwise we ignore that channel because sometimes we do learn some very weird stuff
                        # It is necessary to use this condition because some labels do not necessarily contain all of the classes in one image.
                        effective_classes += 1
                        loss += SoftDiceLoss()(prob_output[:, i, :, :], lbl[:, i, :, :]) + nn.BCELoss(reduction='mean')(prob_output[:, i, :, :].squeeze() + 1e-10, lbl[:, i, :, :].squeeze() + 1e-10)
                        class_outputs = (prob_output[:, i, :, :] > 0.95).float()
                        train_mean_iu_list = segmentation_scores(lbl[:, i, :, :], class_outputs, 2)
                        train_mean_iu_ += sum(train_mean_iu_list) / len(train_mean_iu_list)

                loss = loss / effective_classes
                train_mean_iu_ = train_mean_iu_ / effective_classes
                return {'seg loss': loss,
                        'train iou': train_mean_iu_}

        else:
            print('the output is probably 3D and we do not support it yet')

    else:
        loss = 0.0
        train_mean_iu_ = 0.0
        return {'seg loss': loss,
                'train iou': train_mean_iu_}


def calculate_kl_loss(outputs_dict,
                      b_l,
                      b_u,
                      prior_u,
                      prior_var):
    '''
    Calculate the KL loss between the prior and the posterior for the threshold:
    Args:
        outputs_dict:
        b_l:
        b_u:
        prior_u:
        prior_var:

    Returns:
    '''
    assert b_u > 0
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


def calculate_pseudo_loss(outputs_dict,
                          b_l,
                          b_u,
                          threshold,
                          lung,
                          temp,
                          cutout_aug,
                          apply_lung_mask=True
                          ):
    '''
    Args:
        outputs_dict:
        b_l:
        b_u:
        threshold:
        lung:
        temp:
        cutout_aug:
        apply_lung_mask:
    Returns:
    '''
    assert b_u > 0
    predictions_all = outputs_dict.get('segmentation')
    _, predictions_u = torch.split(predictions_all, [b_l, b_u], dim=0)
    prob_output_u = torch.sigmoid(predictions_u / temp)
    pseudo_label_u = (prob_output_u > threshold).float()

    if apply_lung_mask is True:
        lung_mask = (lung > 0.5)  # float to bool
        prob_output_u = torch.masked_select(predictions_u, lung_mask)
        pseudo_label_u = torch.masked_select(pseudo_label_u, lung_mask)

    if cutout_aug is True:
        cutout = RandomCutOut()
        prob_output_u, pseudo_label_u = cutout.cutout_seg(prob_output_u, pseudo_label_u)

    if len(prob_output_u.size()) == 3:
        # this is binary segmentation
        loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
        return {'pseudo loss': loss}

    elif len(prob_output_u.size()) == 4:
        if prob_output_u.size()[1] == 1:
            # this is also binary segmentation
            loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
            return {'pseudo loss': loss}

        else:
            # this is multi class segmentation
            pseudo_label_u = multi_class_label_processing(pseudo_label_u, prob_output_u.size()[1])  # convert single channel multi integer class label to multi channel binary label
            loss = 0
            effective_classes = 0
            for i in range(prob_output_u.size()[1]):  # multiple
                if torch.sum(pseudo_label_u[:, i, :, :]) > 1.0:
                    # If the channel is not empty, we learn it otherwise we ignore that channel because sometimes we do learn some very weird stuff
                    # It is necessary to use this condition because some labels do not necessarily contain all of the classes in one image.
                    effective_classes += 1
                    loss += SoftDiceLoss()(prob_output_u[:, i, :, :], pseudo_label_u[:, i, :, :]) + nn.BCELoss(reduction='mean')(prob_output_u[:, i, :, :].squeeze() + 1e-10, pseudo_label_u[:, i, :, :].squeeze() + 1e-10)
            loss = loss / effective_classes
            return {'pseudo loss': loss}

    else:
        print('the output is probably 3D and we do not support it yet')


# def train_base(labelled_img,
#                labelled_label,
#                model,
#                unlabelled_img=None,
#                t=2.0,
#                prior_mu=0.4,
#                prior_logsigma=0.1,
#                # augmentation_gaussian=True,
#                # augmentation_zoom=True,
#                # augmentation_contrast=True,
#                augmentation_cutout=True):
def train_base(**kwargs):
    # convert data from numpy to tensor:
    for key, value in kwargs.items():
        inputs = np2tensor_all(img_l=labelled_img, lbl=labelled_label)

        # concatenate labelled and unlabelled for ssl otherwise just use labelled img
        train_img = get_img(inputs)

        # forward pass:
        outputs_dict = model_forward(model, train_img.get('train img'))

        # supervised loss:
        sup_loss = calculate_sup_loss(outputs_dict=outputs_dict,
                                      lbl=inputs.get('lbl'),
                                      temp=t,
                                      b_l=train_img.get('batch labelled'),
                                      b_u=train_img.get('batch unlabelled'),
                                      cutout_aug=augmentation_cutout,
                                      foreground_threshold=50)

        if unlabelled_img is None:
            return {'supervised losses': sup_loss}

        else:
            # calculate the kl and get the learnt threshold:
            kl_loss = calculate_kl_loss(outputs_dict=outputs_dict,
                                        b_l=train_img.get('batch labelled'),
                                        b_u=train_img.get('batch unlabelled'),
                                        prior_u=prior_mu,
                                        prior_var=prior_logsigma)

            # pseudo label loss:
            pseudo_loss = calculate_pseudo_loss(outputs_dict=outputs_dict,
                                                b_l=train_img.get('batch labelled'),
                                                b_u=train_img.get('batch unlabelled'),
                                                threshold=kl_loss.get('threshold unlabelled'),
                                                lung=None,
                                                temp=t,
                                                cutout_aug=augmentation_cutout,
                                                apply_lung_mask=True
                                                )

            return {'supervised losses': sup_loss,
                    'pseudo losses': pseudo_loss,
                    'kl losses': kl_loss}


