# By Ashkan Pakzad ashkanpakzad.github.io
# Script that uses dicom2nifti to convert a folder of dicom cases into nifti. Dicoms should already be anonymised.
# Assumes that each case has a number of dicom series that are seperated into folders and the desired image to convert
# is the one with the most number of slices within its directory.
# 27th February 2021: v0.1.0 initiated

import dicom2nifti
import dicom2nifti.settings as settings
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('inputsource', type=str, help='path to dir holding the dicom dataset')
    parser.add_argument('--outputsource', '-o', default='nii', type=str, help='path to dir to output to nii files')

    return parser

def main(args):

    # dicom2nifti settings
    settings.disable_validate_slice_increment()

    # give parent paths
    parentdicompth = args.inputsource
    parentoutputpth = args.outputsource
    Path(parentoutputpth).mkdir(exist_ok=True)

    # get all cases to execute on
    dcmlist = os.listdir(parentdicompth)
    totaln = len(dcmlist)
    jj = 1
    for case in tqdm(dcmlist):
        jj = jj + 1
        # skip if already converted
        if os.path.exists(os.path.join(parentoutputpth, case+'.nii.gz')):
            continue
        # identify dir with desired dicoms
        casedir = os.path.join(parentdicompth, case)
        subdir = [x[0] for x in os.walk(casedir)]
        Nfiles = [None]*len(subdir)
        for ii in range(len(subdir)):
            Nfiles[ii] = len(os.listdir(subdir[ii]))
        dcmdir = subdir[np.argmax(Nfiles)]

        # convert to nifti without change to orientation
        dicom2nifti.convert_directory(dcmdir, parentoutputpth, compression=True, reorient=False)

        # change name
        alloutputs = os.listdir(parentoutputpth)
        for ii in range(len(alloutputs)):
            if alloutputs[ii].endswith('.nii.gz'):
                alloutputs[ii] = alloutputs[ii].rstrip('.nii.gz')

        # identify cases that are not already converted
        unnamed = list(set(alloutputs) - set(dcmlist))
        if not unnamed:
            print(f'{case} has no valid dicom series. Skipping...')
            continue
        fullunnamed = os.path.join(parentoutputpth, unnamed[0]+'.nii.gz')
        fullnewname = os.path.join(parentoutputpth, case+'.nii.gz')
        os.rename(fullunnamed, fullnewname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('output dicom dataset as nii.gz', parents=[args_parser()])
    args = parser.parse_args()
    main(args)
    print('COMPLETE')
