# Exact Histogram Specification and Histogram Equalization
# Markodimitrakis Manos mtp236
# Based on the project of Tsichlaki Styliani mtp209
# Advances in Digital Imaging and Computer Vision
# env Python 3.7

# libs and imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import skimage
# import imageio
import scipy
import utils
import cv2
import os

# Import Stefano Di Martino's function
# Reference: https://github.com/StefanoD/ExactHistogramSpecification
from .exact_histogram_matching import ExactHistogramMatcher


class HistogramMatcher:

    def __init__(self, output_path, method='ExactHistogramMatching'):
        self.output_path = output_path

        if method == 'ExactHistogramMatching':
            self.histogram_matcher = ExactHistogramMatcher
        else:
            raise ValueError('{v} method is not supported'.format(v=method))

    # Histogram Macthing Function for a batch of images
    def match_histograms(self, target_images, reference_img, using_mask_extraction, display=False):
        if isinstance(target_images, list):
            for image in target_images:
                self.__match_histograms(image, reference_img, using_mask_extraction, display=display)
        else:
            self.__match_histograms(target_images, reference_img, using_mask_extraction, display=display)

    # Histogram Matching Function 
    def __match_histograms(self, target_img, reference_img, using_mask_extraction=False, display=False):

        target_img_path = None
        target_img_name = None

        # Checking if the value of the variable is a filepath or an image array 
        # Read Target Image from the path
        if isinstance(target_img, str):
            target_img_path = target_img.split('/')[-2]
            target_img_name = target_img.split('/')[-1]
            reference_img_name = reference_img.split('/')[-1]

            target_image = skimage.io.imread(target_img)
        # Do nothing if variable is an image
        elif isinstance(target_img, np.ndarray):
            pass
        else:
            raise TypeError("Unkown file type: %{}".format(type(target_img)))

        # Checking if the value of the variable is a filepath or an image array 
        # Read Reference Image from the path
        if isinstance(reference_img, str):
            reference_image = skimage.io.imread(reference_img)
        # Do nothing if variable is an image
        elif isinstance(reference_img, np.ndarray):
            pass
        else:
            raise TypeError("Unkown file type: %{}".format(type(reference_img)))

        if len(target_image.shape) != len(reference_image.shape):
            raise ValueError("Target image shape must be the same as the reference image shape")  # TODO: is this right?

        # Find masks from image paths
        target_img_mask_path = target_img.split('/')[:-1]
        target_img_mask_path = '/'.join(target_img_mask_path)
        img_name = target_img_name.split('_')[:-1]
        img_name = '_'.join(img_name)
        target_img_mask_path = target_img_mask_path + '/' + img_name + "_mask.nii"

        reference_img_mask_path = reference_img.split('/')[:-1]
        reference_img_mask_path = '/'.join(reference_img_mask_path)
        img_name = reference_img_name.split('_')[:-1]
        img_name = '_'.join(img_name)
        reference_img_mask_path = reference_img_mask_path + '/' + img_name + "_mask.nii"

        if not os.path.isfile(target_img_mask_path):
            raise FileNotFoundError("File " + target_img_mask_path + " not found!")

        if not os.path.isfile(reference_img_mask_path):
            raise FileNotFoundError("File " + reference_img_mask_path + " not found!")

        # Read masks
        target_image_mask = skimage.io.imread(target_img_mask_path)
        reference_image_mask = skimage.io.imread(reference_img_mask_path)

        # Histogram Equalization to target image
        if len(target_image.shape) == 3:
            if using_mask_extraction:
                target_image = utils.remove_mask_from_image(target_image, target_image_mask)
            target_image_equalized = utils.histogram_equalization_3D(target_image)
        else:
            target_image_equalized = utils.histogram_equalization_2D(target_img)

        # target_image_equalized = utils.histogram_equalization_CLAHE(target_img)

        # Histogram Equalization to reference image
        if len(reference_image.shape) == 3:
            if using_mask_extraction:
                reference_image = utils.remove_mask_from_image(reference_image, reference_image_mask)
            reference_image_equalized = utils.histogram_equalization_3D(target_image)
        else:
            reference_image_equalized = utils.histogram_equalization_2D(reference_image)

        # reference_image_equalized = utils.histogram_equalization_CLAHE(reference_img)

        # Find the histogram of the reference image
        reference_histogram = self.histogram_matcher.get_histogram(reference_image_equalized)

        # Match target image to the reference histogram
        hist_matched_img = self.histogram_matcher.match_image_to_histogram(target_image_equalized, reference_histogram)

        # Result image
        hist_matched_img = np.uint8(hist_matched_img)

        new_dir = os.path.join(self.output_path, target_img_path)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

        correct_headers_path = os.path.join('data', 'correct_headers')
        correct_headers_nifti_path = os.path.join(correct_headers_path, 'img_with_correct_header.nii')
        img_with_correct_header = nib.load(correct_headers_nifti_path)
        affine = img_with_correct_header.affine
        # header = img_with_correct_header.header
        hist_matched_img = np.rot90(hist_matched_img)
        hist_matched_img = np.flipud(hist_matched_img)

        nib.Nifti1Image(hist_matched_img, affine).to_filename(os.path.join(new_dir, target_img_name))
        # skimage.io.imsave(os.path.join(new_dir, target_img_name), hist_matched_img)

        if display:
            # Plot
            # Pseudocolor used, which is only relevant to single-channel, grayscale, luminosity images.
            # It can be a useful tool for enhancing contrast and visualizing data more easily
            figure3 = plt.figure()

            # Subplot Layout
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

            # Target Image
            subplot = figure3.add_subplot(321)
            subplot.set_title('Target Image Equalized')
            plt.imshow(np.array(target_image_equalized[:, :, 0], np.int32), cmap='gray')

            # Target Image Histogram
            subplot = figure3.add_subplot(322)
            subplot.set_title('Target Image Histogram')
            plt.hist(target_image_equalized.flatten(), 256, [0, 256])

            # Reference Image
            subplot = figure3.add_subplot(323)
            subplot.set_title('Reference Image Equalized')
            plt.imshow(np.array(reference_image_equalized[:, :, 0], np.int32), cmap='gray')

            # Reference Image Histogram
            subplot = figure3.add_subplot(324)
            subplot.set_title('Reference Image Histogram')
            plt.hist(reference_image_equalized.flatten(), 256, [0, 256])

            # Result Image (Matched to Histogram)
            subplot = figure3.add_subplot(325)
            hist_matched_img = np.rot90(hist_matched_img)
            hist_matched_img = np.flipud(hist_matched_img)
            plt.imshow(np.array(hist_matched_img[:, :, 0], np.int32), cmap='gray')

            subplot.set_title('Image Matched to Histogram')

            # Result image's Histogram
            subplot = figure3.add_subplot(326)
            subplot.set_title('Result Image Histogram')
            plt.hist(hist_matched_img.flatten(), 256, [0, 256])
            plt.xlim([0, 256])
            plt.show()
