import torch
import torch.nn as nn
from libs.Loss import SoftDiceLoss
from libs.Metrics import segmentation_scores


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


def check_inputs(**kwargs):
    outputs = {}
    for key, val in kwargs.items():
        # check the dimension for each input
        outputs[key] = check_dim(val)
    return outputs


def np2tensor_all(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = {}
    for key, val in kwargs.items():
        outputs[key] = val.to(device=device, dtype=torch.float32)
    outputs = check_inputs(**outputs)
    return outputs


def get_img(**inputs):
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


def calculate_sup_loss(lbl,
                       outputs_dict,
                       temp):

    raw_output = outputs_dict.get('segmentation')
    _, pseudo_label = torch.max(raw_output, dim=1)
    prob_output = torch.softmax(raw_output / temp, dim=1)

    # mask_labelled = torch.zeros_like(lbl).cuda()
    # mask_unlabelled = torch.zeros_like(lbl).cuda()
    # mask_labelled[lbl > 0] = 1
    # mask_unlabelled[lbl == 0] = 1
    # mask_labelled = lbl.ge(0.5)

    # mask the labels:
    # prob_output_foreground = prob_output*mask_labelled
    # lbl_foreground = lbl*mask_labelled
    # raw_output_foreground = raw_output*mask_labelled
    # pseudo_label_foreground = pseudo_label*mask_labelled

    # for labelled parts:
    # loss_sup = SoftDiceLoss()(prob_output_foreground, lbl_foreground) + nn.CrossEntropyLoss(reduction='mean')(raw_output_foreground, lbl_foreground.long())
    loss_sup = SoftDiceLoss()(prob_output, lbl) + nn.CrossEntropyLoss(reduction='mean')(raw_output, lbl.long())

    # # for unlabelled parts:
    # # (todo) fix this
    # loss_unsup = SoftDiceLoss()(prob_output, pseudo_label, mask_unlabelled) + nn.CrossEntropyLoss(reduction='mean')(output*mask_unlabelled / temp, pseudo_label.long()*mask_unlabelled.long())

    # training iou
    # train_mean_iu_ = segmentation_scores(lbl_foreground, pseudo_label_foreground, prob_output.size()[1])
    train_mean_iu_ = segmentation_scores(lbl, pseudo_label, prob_output.size()[1])

    return {'loss_sup': loss_sup,
            'train iou': train_mean_iu_}


def train_sup(labelled_img,
              labelled_label,
              model,
              t=2.0):

    inputs = np2tensor_all(**{'img_l': labelled_img, 'lbl': labelled_label})
    train_img = get_img(**inputs)

    # Check if the input is 3D:
    if len(train_img.get('train img').size()) == 3: # 2D
        train_img_data = train_img.get('train img').unsqueeze(1)
    elif len(train_img.get('train img').size()) == 4: # 3D
        train_img_data = train_img.get('train img').unsqueeze(1)
    else:
        raise NotImplementedError

    outputs_dict = model_forward(model, train_img_data)
    sup_loss = calculate_sup_loss(outputs_dict=outputs_dict,
                                  lbl=inputs.get('lbl'),
                                  temp=t)

    return {'supervised losses': sup_loss}

    # if torch.sum(lbl) > 10: # check whether there are enough foreground pixels
    #     output = outputs_dict.get('segmentation') # get segmentation map
    #     if b_u: # check if unlabelled data is included
    #         output, _ = torch.split(output, [b_l, b_u], dim=0) # if both labelled and unlabelled, split the data and use the labelled
    #
    #     prob_output = torch.softmax(output, dim=1) # Apply softmax function along the dimension
    #
    #     if cutout_aug == 1: # apply cutout augmentation
    #         prob_output, lbl = randomcutout(prob_output, lbl)
    #
    #     # this is also binary segmentation
    #     mask = torch.zeros_like(lbl).cuda()
    #     mask[lbl > 0] = 1 # use only labelled regions
    #     loss = SoftDiceLoss()(prob_output, lbl, mask) + nn.CrossEntropyLoss(reduction='mean')(prob_output.squeeze()*mask, lbl.squeeze()*mask)
    #     class_outputs = (prob_output > 0.95).float()
    #     train_mean_iu_ = segmentation_scores(lbl*mask, class_outputs*mask, prob_output.size()[1])
    #     return {'loss': loss.mean(),
    #             'train iou': train_mean_iu_}
    #
    # else:
    #     loss = torch.tensor(0.0).to('cuda')
    #     train_mean_iu_ = 0.0
    #     return {'loss': loss,
    #             'train iou': train_mean_iu_}

#
# def calculate_kl_loss(outputs_dict,
#                       b_u,
#                       b_l,
#                       prior_u,
#                       # prior_var
#                       ):
#
#     assert b_u > 0
#     posterior_mu = outputs_dict.get('mu')
#     posterior_logvar = outputs_dict.get('logvar')
#
#     confidence_threshold_learnt = outputs_dict.get('learnt_threshold')
#
#     # loss = kld_loss(posterior_mu, posterior_logvar, prior_u, prior_var)
#     loss = kld_loss(posterior_mu, posterior_logvar, prior_u)
#
#     threshold_learnt_l, threshold_learnt_u = torch.split(confidence_threshold_learnt, [b_l, b_u], dim=0)
#
#     return {'loss': loss.mean(),
#             'threshold labelled': threshold_learnt_l,
#             'threshold unlabelled': threshold_learnt_u}


# def calculate_pseudo_loss(outputs_dict,
#                           b_u,
#                           b_l,
#                           temp,
#                           lbl,
#                           cutout_aug=0,
#                           conf_threshold='bayesian'):
#
#     assert b_u > 0
#     predictions_all = outputs_dict.get('segmentation')
#
#     # Monte Carlo sampling of confidence threshold:
#     if conf_threshold == 'bayesian':
#         threshold = outputs_dict.get('learnt_threshold')
#     else:
#         threshold = 0.5
#
#     predictions_l, predictions_u = torch.split(predictions_all, [b_l, b_u], dim=0)
#     threshold_l, threshold_u = torch.split(threshold, [b_l, b_u], dim=0)
#     prob_output_u = torch.softmax(predictions_u / temp, dim=1)
#     pseudo_label_u = (prob_output_u >= threshold_u).float()
#     prob_output_l = torch.softmax(predictions_l / temp, dim=1)
#     pseudo_label_l = (prob_output_l >= threshold_l).float()
#
#     if cutout_aug == 1:
#         prob_output_u, pseudo_label_u = randomcutout(prob_output_u, pseudo_label_u)
#
#     mask = torch.zeros_like(lbl)

    # if torch.sum(pseudo_label_u) > 10:
    #     if len(prob_output_u.size()) == 3:
    #         # this is binary segmentation
    #         loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
    #         loss += 0.5*SoftDiceLoss()(prob_output_l, pseudo_label_l) + nn.BCELoss(reduction='mean')(prob_output_l.squeeze() + 1e-10, pseudo_label_l.squeeze() + 1e-10)
    #         return {'loss': loss.mean()}
    #
    #     elif len(prob_output_u.size()) == 4:
    #         if prob_output_u.size()[1] == 1:
    #             # this is also binary segmentation
    #             loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
    #             loss += 0.5*SoftDiceLoss()(prob_output_l, pseudo_label_l) + nn.BCELoss(reduction='mean')(prob_output_l.squeeze() + 1e-10, pseudo_label_l.squeeze() + 1e-10)
    #             return {'loss': loss.mean()}
    #
    #         else:
    #             # this is multi class segmentation
    #             pseudo_label_u = multi_class_label_processing(pseudo_label_u, prob_output_u.size()[1])  # convert single channel multi integer class label to multi channel binary label
    #             loss = torch.tensor(0).to('cuda')
    #             effective_classes = 0
    #             for i in range(prob_output_u.size()[1]):  # multiple
    #                 if torch.sum(pseudo_label_u[:, i, :, :]) > 10.0:
    #                     # If the channel is not empty, we learn it otherwise we ignore that channel because sometimes we do learn some very weird stuff
    #                     # It is necessary to use this condition because some labels do not necessarily contain all of the classes in one image.
    #                     effective_classes += 1
    #                     loss += SoftDiceLoss()(prob_output_u[:, i, :, :], pseudo_label_u[:, i, :, :]).mean() + nn.BCELoss(reduction='mean')(prob_output_u[:, i, :, :].squeeze() + 1e-10, pseudo_label_u[:, i, :, :].squeeze() + 1e-10).mean()
    #                     loss += 0.5*SoftDiceLoss()(prob_output_l[:, i, :, :], pseudo_label_l[:, i, :, :]).mean() + nn.BCELoss(reduction='mean')(prob_output_l[:, i, :, :].squeeze() + 1e-10, pseudo_label_l[:, i, :, :].squeeze() + 1e-10).mean()
    #             loss = loss / effective_classes
    #             return {'loss': loss.mean()}

    # else:
    #     return {'loss': torch.tensor(0.0).to('cuda').mean()}


# b_l = train_img.get('batch labelled'),
# b_u = train_img.get('batch unlabelled'),

# def train_semi(labelled_img,
#                labelled_label,
#                model,
#                unlabelled_img,
#                t=2.0,
#                prior_mu=0.7,
#                # prior_logsigma=0.1,
#                augmentation_cutout=0):
#
#     # convert data from numpy to tensor:
#     inputs = np2tensor_all(**{'img_l': labelled_img,
#                               'lbl': labelled_label,
#                               'img_u': unlabelled_img})
#
#     # concatenate labelled and unlabelled for ssl otherwise just use labelled img
#     train_img = get_img(**inputs)
#
#     # forward pass:
#     outputs_dict = model_forward(model, train_img.get('train img'))
#
#     # supervised loss:
#     sup_loss = calculate_sup_loss(outputs_dict=outputs_dict,
#                                   lbl=inputs.get('lbl'),
#                                   temp=t,
#                                   b_l=train_img.get('batch labelled'),
#                                   b_u=train_img.get('batch unlabelled'),
#                                   cutout_aug=augmentation_cutout)
#
#     # calculate the kl and get the learnt threshold:
#     kl_loss = calculate_kl_loss(outputs_dict=outputs_dict,
#                                 b_l=train_img.get('batch labelled'),
#                                 b_u=train_img.get('batch unlabelled'),
#                                 prior_u=prior_mu,
#                                 # prior_var=prior_logsigma
#                                 )
#
#     # pseudo label loss:
#     pseudo_loss = calculate_pseudo_loss(outputs_dict=outputs_dict,
#                                         b_l=train_img.get('batch labelled'),
#                                         b_u=train_img.get('batch unlabelled'),
#                                         temp=t,
#                                         cutout_aug=augmentation_cutout
#                                         )
#
#     return {'supervised loss': sup_loss,
#             'pseudo loss': pseudo_loss,
#             'kl loss': kl_loss}

