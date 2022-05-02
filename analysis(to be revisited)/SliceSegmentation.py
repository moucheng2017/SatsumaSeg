import torch
import random
import numpy as np
import os

import glob
import tifffile as tiff

import errno
import imageio

from PIL import Image
from torch.utils import data

import matplotlib.pyplot as plt

from Utils import CustomDataset

# class CustomDataset(torch.utils.data.Dataset):
#
#     def __init__(self, imgs_folder, labels_folder, augmentation):
#
#         # 1. Initialize file paths or a list of file names.
#         self.imgs_folder = imgs_folder
#         self.labels_folder = labels_folder
#         self.data_augmentation = augmentation
#         # self.transform = transforms
#
#     def __getitem__(self, index):
#         # 1. Read one data from file (e.g. using num py.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#
#         all_images = glob.glob(os.path.join(self.imgs_folder, '*.png'))
#         all_labels = glob.glob(os.path.join(self.labels_folder, '*.tif'))
#         # sort all in the same order
#         all_labels.sort()
#         all_images.sort()
#
#         # label = Image.open(all_labels[index])
#         label = tiff.imread(all_labels[index])
#         label_origin = np.array(label, dtype='float32')
#
#         image = Image.open(all_images[index])
#         # image = tiff.imread(all_images[index])
#         image = np.array(image, dtype='float32')
#
#         labelname = all_labels[index]
#         path_label, labelname = os.path.split(labelname)
#         labelname, labelext = os.path.splitext(labelname)
#
#         c_amount = len(np.shape(label))
#
#         # Reshaping everyting to make sure the order: channel x height x width
#         if c_amount == 3:
#             d1, d2, d3 = np.shape(label)
#
#             if d1 != min(d1, d2, d3):
#
#                 label = np.reshape(label, (d3, d1, d2))
#                 c = d3
#                 h = d1
#                 w = d2
#             else:
#                 c = d1
#                 h = d2
#                 w = d3
#
#         elif c_amount == 2:
#             h, w = np.shape(label)
#             label = np.reshape(label_origin, (1, h, w))
#
#         d1, d2, d3 = np.shape(image)
#         if d1 != min(d1, d2, d3):
#
#             image = np.reshape(image, (d3, d1, d2))
#
#         return image, label, labelname
#
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return len(glob.glob(os.path.join(self.labels_folder, '*.tif')))


# ====================================================================================
# ====================================================================================
def plot(model, modelname, save_location, data):
    #
    device = torch.device('cuda')
    #
    model = model.to(device=device)
    #
    model.eval()
    #
    with torch.no_grad():
        #
        evaluate_index_all = range(0, len(data))
        # evaluate_index_all = range(0, 100)
        #
        for index in evaluate_index_all:
            # extract a few random indexs every time in a range of the data
            testimg, testlabel, test_imagename = data[index]
            # print(test_imagename)
            #
            # augmentation = random.random()
            # #
            # if augmentation > 0.5:
            #     c, h, w = np.shape(testimg)
            #     for channel in range(c):
            #         testimg[channel, :, :] = np.flip(testimg[channel, :, :], axis=0).copy()
            #
            # ========================================================================
            # ========================================================================
            testimg = torch.from_numpy(testimg).to(device=device, dtype=torch.float32)
            testlabel = torch.from_numpy(testlabel).to(device=device, dtype=torch.float32)

            if torch.max(testlabel) == 255.:
                testlabel = testlabel / 255.

            c, h, w = testimg.size()

            # if c > 1:
            #     testimg = testimg[0, :, :].view(1, h, w).contiguous()
            #     testimg = testimg.expand(1, 1, h, w)
            # else:

            testimg = testimg.expand(1, c, h, w)

            testoutput_original = model(testimg)
            testoutput = torch.sigmoid(testoutput_original.view(1, h, w))

            threshold = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=False)
            upper = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=False)
            lower = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)

            testoutput = torch.where(testoutput > threshold, upper, lower)

            try:
                os.mkdir(save_location)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            # Plot error maps:
            # testimg = testimg[:, 0, :, :]
            # testimg = testimg[:, 0, :, :]
            testimg = testimg.cpu().squeeze().detach().numpy()
            testimg = np.asarray(testimg, dtype=np.float32)
            testimg = testimg.transpose((1, 2, 0))

            # u, count = np.unique(testimg, return_counts=True)
            # print(np.asarray((u, count)).T)

            c_l, h_l, w_l = testlabel.size()

            if c_l > 1:
                testlabel = testlabel[:, 0, :, :]
            else:
                pass

            label = testlabel.squeeze().cpu().detach().numpy()
            pred = testoutput.squeeze().cpu().detach().numpy()

            label = np.asarray(label, dtype=np.uint8)
            pred = np.asarray(pred, dtype=np.uint8)

            error_map = np.zeros((h, w, 3), dtype=np.uint8)

            gt_map = np.zeros((h, w, 3), dtype=np.uint8)

            # error_map[difference == -1] = [255, 0, 0]  # false positive red
            # error_map[difference == 1] = [0, 0, 255]  # false negative blue
            # error_map[addition == 0] = [0, 255, 0]  # true negative green
            # error_map[addition == 2] = [255, 255, 0]  # true positive yellow

            # error_map[label == 0 & pred == 1] = [255, 0, 0]  # false positive red
            # error_map[label == 1 & pred == 0] = [0, 0, 255]  # false negative blue
            # error_map[label == 0 & pred == 0] = [0, 255, 0]  # true negative green
            # error_map[label == 1 & pred == 1] = [255, 255, 0]  # true positive yellow

            for hh in range(0, h):
                for ww in range(0, w):
                    label_ = label[hh, ww]
                    pred_ = pred[hh, ww]
                    # pixel = testimg[hh, ww, :]
                    if label_ == 1 and pred_ == 1:
                        error_map[hh, ww, 0] = 255
                        error_map[hh, ww, 1] = 255
                        error_map[hh, ww, 2] = 0
                    elif label_ == 0 and pred_ == 0:
                        error_map[hh, ww, 0] = 0
                        error_map[hh, ww, 1] = 255
                        error_map[hh, ww, 2] = 0
                    elif label_ == 0 and pred_ == 1:
                        error_map[hh, ww, 0] = 255
                        error_map[hh, ww, 1] = 0
                        error_map[hh, ww, 2] = 0
                    elif label_ == 1 and pred_ == 0:
                        error_map[hh, ww, 0] = 0
                        error_map[hh, ww, 1] = 0
                        error_map[hh, ww, 2] = 255

                    # if pixel < -1:
                    #     error_map[hh, ww, :] = 0

                    if label_ == 1:
                        gt_map[hh, ww, 0] = 255
                        gt_map[hh, ww, 1] = 255
                        gt_map[hh, ww, 2] = 0

            prediction_name = 'seg_' + test_imagename + '.png'
            full_error_map_name = os.path.join(save_location, prediction_name)
            imageio.imsave(full_error_map_name, error_map)

            pic_name = 'original_' + test_imagename + '.png'
            full_pic_map_name = os.path.join(save_location, pic_name)

            gt_name = 'gt_' + test_imagename + '.png'
            full_gt_map_name = os.path.join(save_location, gt_name)
            imageio.imsave(full_gt_map_name, gt_map)

            # check testimg shape:
            # c, h, w = testimg.shape
            # if c == 3:
            #     testimg = np.reshape(testimg, (h, w, c))

            imageio.imsave(full_pic_map_name, testimg)

            seg_img = Image.open(full_error_map_name)
            input_img = Image.open(full_pic_map_name)

            seg_img = seg_img.convert("RGBA")
            input_img = input_img.convert("RGBA")

            alphaBlended = Image.blend(seg_img, input_img, alpha=.6)
            blended_name = 'blend_' + test_imagename + '.png'
            full_blend_map_name = os.path.join(save_location, blended_name)
            imageio.imsave(full_blend_map_name, alphaBlended)

            gt_img = Image.open(full_gt_map_name)
            input_img = Image.open(full_pic_map_name)

            gt_img = gt_img.convert("RGBA")
            input_img = input_img.convert("RGBA")

            betaBlended = Image.blend(gt_img, input_img, alpha=.6)
            imageio.imsave(full_gt_map_name, betaBlended)

            # os.remove(full_pic_map_name)
            # os.remove(full_error_map_name)

            print(full_error_map_name + ' is saved.')


# ===============================================================
if __name__ == '__main__':

    name1 = 'epoch91.pt'
    # name2 = 'fcn.pt'
    # name3 = 'segnet.pt'
    # name4 = 'unet.pt'
    # name5 = 'aunet.pt'
    # name6 = 'maunet.pt'
    # name7 = 'uraunet.pt'

    model_names = [name1]

    for model_name in model_names:
        print(model_name)
        path = '/home/moucheng/neuroblastoma/UNet_batch_2_width_32_repeat_1_augment_full_lr_decay_True_data_e100_lr0.001/trained_models/' + model_name
        print(path)
        # model = torch.model(path)
        model = torch.load(path)
        print('Model is loaded.')
        # ============================
        # There are issues with models trained with different versions of libraries.
        # models trained on cluster can't do inferences locally... different versions in libraries
        # To solve this:
        # 1. call a new empty model of testing model
        # 2. load the weights in the new emtpy model with weights from testing model
        # 3. load the new model, this avoids the differences of versions of libraies
        # ============================
        # model_dict = model.state_dict()
        # # print(model_dict)
        # # new_model = AttentionUNet(in_ch=3, width=16, visulisation=True)
        # # new_model = FPAttentionUNet(in_ch=3, width=16, fpa_type='2', dilation=6, attention_no=3, width_expansion=1, param_share_times=2, visulisation=True)
        # new_model = FNAttentionUNet(in_ch=3, width=16, attention_no=3, width_expansion=4, attention_type='2', visulisation=True)
        # new_model_dict = new_model.state_dict()
        # model_dict = {k: v for k, v in model_dict.items() if k in new_model_dict.keys()}
        # new_model_dict.update(model_dict)
        # new_path = '/home/moucheng/Results/cityscape/models/new_' + model_name
        # torch.save(new_model, new_path)
        # #
        # # model = new_model.load_state_dict(torch.load(new_path))
        # model = torch.load(new_path)
        # =====================================================================
        data_folder = '/home/moucheng/neuroblastoma/'
        # validate_image_folder = data_folder + str(fold_no) + '/validate/patches'
        # validate_label_folder = data_folder + str(fold_no) + '/validate/labels'
        validate_image_folder = data_folder + '/test_data/patches'
        validate_label_folder = data_folder + '/test_data/labels'
        data = CustomDataset(validate_image_folder, validate_label_folder, False)

        save_location = '/home/moucheng/neuroblastoma/test_results'

        try:
            os.mkdir(save_location)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        #
        save_location = save_location + '/' + model_name[: -3]
        #
        plot(model=model, save_location=save_location, data=data, modelname=model_name)
    #
    # fold_no = 2
    # for model_name in model_names:
    #     path = '/home/moucheng/projects_codes/saved_models/' + model_name
    #     model = torch.load(path)
    #     print('Model is loaded.')
    #     data_folder = '/home/moucheng/projects_data/Brain_data/BRATS2018/MICCAI_BraTS_2018_Data_Training/ET_L0_H210_f5/'
    #     validate_image_folder = data_folder + str(fold_no) + '/validate/patches'
    #     validate_label_folder = data_folder + str(fold_no) + '/validate/labels'
    #     data = CustomDataset(validate_image_folder, validate_label_folder, False)
    #     #
    #     save_location = '/home/moucheng/projects_data/Segmentation/fold_' + str(fold_no)
    #     #
    #     try:
    #         os.mkdir(save_location)
    #     except OSError as exc:
    #         if exc.errno != errno.EEXIST:
    #             raise
    #         pass
    #     #
    #     save_location = save_location + '/' + model_name[: -3]
    #     #
    #     plot(model=model, save_location=save_location, data=data, modelname=model_name)
    # #
    # fold_no = 4
    # for model_name in model_names:
    #     path = '/home/moucheng/projects_codes/saved_models/' + model_name
    #     model = torch.load(path)
    #     print('Model is loaded.')
    #     data_folder = '/home/moucheng/projects_data/Brain_data/BRATS2018/MICCAI_BraTS_2018_Data_Training/ET_L0_H210_f5/'
    #     validate_image_folder = data_folder + str(fold_no) + '/validate/patches'
    #     validate_label_folder = data_folder + str(fold_no) + '/validate/labels'
    #     data = CustomDataset(validate_image_folder, validate_label_folder, False)
    #     #
    #     save_location = '/home/moucheng/projects_data/Segmentation/fold_' + str(fold_no)
    #     #
    #     try:
    #         os.mkdir(save_location)
    #     except OSError as exc:
    #         if exc.errno != errno.EEXIST:
    #             raise
    #         pass
    #     #
    #     save_location = save_location + '/' + model_name[: -3]
    #     #
    #     plot(model=model, save_location=save_location, data=data, modelname=model_name)
    #     #
print('End')
