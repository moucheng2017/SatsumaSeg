import os
import numpy as np


if __name__ == '__main__':

    # cut chunks along d dimension into cubes every 162 pixels
    img_path = '/home/moucheng/projects_data/hipct_covid/original_validate/imgs'
    lbl_path = '/home/moucheng/projects_data/hipct_covid/original_validate/lbls'

    save_path_img = '/home/moucheng/projects_data/hipct_covid/validate/imgs/'
    save_path_lbl = '/home/moucheng/projects_data/hipct_covid/validate/lbls/'

    imgs = os.listdir(img_path)
    lbls = os.listdir(lbl_path)
    imgs = [os.path.join(img_path, i) for i in imgs]
    lbls = [os.path.join(lbl_path, i) for i in lbls]

    count = 0
    for img, lbl in zip(imgs, lbls):
        img = np.load(img)
        lbl = np.load(lbl)
        img = np.asfarray(img)
        lbl = np.asfarray(lbl)
        lbl[lbl != 1.0] = 0
        print(np.shape(img))
        d, h, w = np.shape(img)
        sub_imgs = np.split(img, d // 160, axis=0)
        sub_lbls = np.split(lbl, d // 160, axis=0)
        for each_sub_img, each_sub_lbl in zip(sub_imgs, sub_lbls):
            np.save(save_path_img+str(count)+'img.npy', each_sub_img)
            np.save(save_path_lbl+str(count)+'lbl.npy', each_sub_lbl)
            count += 1
    print('Done')

