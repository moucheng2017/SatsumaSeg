# pip install git+https://github.com/JoHof/lungmask

from lungmask import mask
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from tqdm import tqdm

def main():
    # get all lung rawimage paths
    images_dir = Path('/home/moucheng/projects_data/Pulmonary_data/airway/Mixed/test/imgs')
    image_paths = sorted(images_dir.glob('*.nii.gz'))

    label_dir = Path('/home/moucheng/projects_codes/Results/cluster/Results/airway/Mixed/20200206/sup_unet_e1_l0.0001_b2_w16_s4000_d4_r0.05_z16_x384/segmentation/lung_label')


    # load rawimage
    for i, path in enumerate(image_paths):
        print(f'starting {i} of {len(image_paths)}')
        try:
            input_image = sitk.ReadImage(str(path))
            # apply model
            result = mask.apply(input_image)
            # make binary segmentation
            result_processed = result.copy()
            result_processed[result > 1] = 1
            # convert to itk
            result_itk = sitk.GetImageFromArray(result_processed)
            result_itk.CopyInformation(input_image)
            # save result
            seg_path_name = label_dir / (path.stem[:-4] + '_lunglabel.nii.gz')
            sitk.WriteImage(result_itk, str(seg_path_name))
        except RuntimeError:
            print(f'Fail :{path}')


if __name__ == '__main__':
    main()
    print('COMPLETE lung label gen')
