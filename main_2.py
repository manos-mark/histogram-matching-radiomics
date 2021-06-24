import glob

import utils
import os

DATASET_PATH = os.path.join('data', 'dataset')


def getSingleImageNii(dataset_path, contrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, '*_' + contrast_type + '.tif'))

        if not filenames:
            raise ValueError(f'dir ({dir}) does not contain any .tif or .tiff images.')

        utils.single_tiff_to_nii(filenames, dir, contrast_type)


getSingleImageNii(DATASET_PATH, 'post-contrast')
