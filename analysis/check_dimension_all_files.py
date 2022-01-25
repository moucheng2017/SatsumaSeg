import glob
import os
# import gzip
# import shutil
# import random
import errno
import numpy as np


def read_all_files(path):
    all_files = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            all_files.append(os.path.join(path, name))
            # print(os.path.join(path, name))
    return all_files


# def separate_files(all_files_path, save_root_path):
#     path_for_each_resolution = {}


if __name__ == '__main__':

    allfiles = read_all_files('/home/moucheng/projects_data/Pulmonary_data/airway')
    dim1 = 0
    dim2 = 0

    dim1_paths = []
    dim2_paths = []

    for file in allfiles:
        data = np.load(file)
        # print(np.shape(file))
        c, d, h, w = np.shape(data)
        if h == 512:
            dim1+=1
            dim1_paths.append(file)
        elif h == 768:
            dim2+=1
            dim2_paths.append(file)

    print('cases of 512:' + str(dim1//2))
    for file in dim1_paths:
        print(file)

    print('\n')
    print('\n')

    print('case of 768:' + str(dim2//2))
    for file in dim2_paths:
        print(file)