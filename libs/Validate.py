import random

import torch
from Metrics import segmentation_scores


def validate_base(val_img,
                  val_lbl,
                  val_lung,
                  device,
                  model,
                  classes=2,
                  threshold=0.95,
                  apply_lung_mask=True):
        '''

        Args:
            val_img:
            val_lbl:
            val_lung:
            device:
            model:
            classes:
            threshold:
            apply_lung_mask:

        Returns:

        '''
        val_img = val_img.to(device, dtype=torch.float32)
        val_lbl = val_lbl.to(device, dtype=torch.float32)
        val_lung = val_lung.to(device, dtype=torch.float32)

        # forward pass:
        outputs_dict = model(val_img)
        val_output = outputs_dict.get('segmentation')

        val_output = torch.sigmoid(val_output)
        val_output = (val_output >= threshold).float()

        lung_mask = (val_lung > 0.5) # make lung mask into bool
        val_class_outputs_masked = torch.masked_select(val_output, lung_mask)
        val_label_masked = torch.masked_select(val_lbl, lung_mask)
        if apply_lung_mask is True:
            eval_mean_iu_ = segmentation_scores(val_label_masked.squeeze(), val_class_outputs_masked.squeeze(), classes)
        else:
            eval_mean_iu_ = segmentation_scores(val_lbl.squeeze(), val_output.squeeze(), classes)
        return eval_mean_iu_


def validate_three_planes(validate_loader,
                          device,
                          model,
                          no_validate=10,
                          full_orthogonal=True):
    '''

    Args:
        validate_loader:
        device:
        model:

    Returns:

    '''
    model.eval()
    with torch.no_grad():
        if full_orthogonal is True:
            val_iou_d = []
            val_iou_h = []
            val_iou_w = []
            iterator_val_labelled = iter(validate_loader)
            for i in range(no_validate):
                try:
                    val_dict, _ = next(iterator_val_labelled)
                except StopIteration:
                    iterator_val_labelled = iter(validate_loader)
                    val_dict, _ = next(iterator_val_labelled)

                val_iou_d.append(validate_base(val_dict["plane_d"][0], val_dict["plane_d"][1], val_dict["plane_d"][2], device, model))
                val_iou_h.append(validate_base(val_dict["plane_h"][0], val_dict["plane_h"][1], val_dict["plane_h"][2], device, model))
                val_iou_w.append(validate_base(val_dict["plane_w"][0], val_dict["plane_w"][1], val_dict["plane_w"][2], device, model))

                return {"val d plane": val_iou_d,
                        "val h plane": val_iou_h,
                        "val w plane": val_iou_w}

        else:
            val_iou = []
            iterator_val_labelled = iter(validate_loader)
            for i in range(no_validate):
                try:
                    val_dict, _ = next(iterator_val_labelled)
                except StopIteration:
                    iterator_val_labelled = iter(validate_loader)
                    val_dict, _ = next(iterator_val_labelled)

                if random.random() < 0.34:
                    val_iou.append(validate_base(val_dict["plane_d"][0], val_dict["plane_d"][1], val_dict["plane_d"][2], device, model))
                elif random.random() < 0.68:
                    val_iou.append(validate_base(val_dict["plane_h"][0], val_dict["plane_h"][1], val_dict["plane_h"][2], device, model))
                else:
                    val_iou.append(validate_base(val_dict["plane_w"][0], val_dict["plane_w"][1], val_dict["plane_w"][2], device, model))

            return {"val plane": val_iou}
