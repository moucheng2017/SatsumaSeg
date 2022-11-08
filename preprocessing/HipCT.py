import os
import pathlib
import numpy as np
from PIL import Image
from collections import deque


# check the dimension of data input:
def check_vol_dim(path):
    all_files = os.listdir(path)
    all_files = [os.path.join(path, f) for f in all_files if 'stack_' in f]
    for f in all_files:
        filename = pathlib.Path(f).stem
        dim = np.shape(np.load(f))
        # print file name and dimensions:
        print(filename + ' : %d , %d , %d' % (dim[0], dim[1], dim[2]))


def sample_more_unlabelled_volumes(path):
    original_sample = Image.open(path)
    original_sample = np.array(original_sample)
    print(np.shape(original_sample))


if __name__ == '__main__':
    # check_vol_dim('/home/moucheng/projects_data/HipCT/COVID_ML_data/COVID-CNN/paired_dataset')
    path='/home/moucheng/projects_data/HipCT/COVID_ML_data/original_data_full_stacks/Substack (1500-2500)6.24um_FO-20.129-OL_column4_pag-0.02_0.06_bin2_.tif'
    sample_more_unlabelled_volumes(path)