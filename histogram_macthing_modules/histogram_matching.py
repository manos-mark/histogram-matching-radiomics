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
    def perform_batch_histogram_matching(self, target_images, reference_img, display=False):
        for image in target_images:
            self.perform_histogram_matching(image, reference_img, display=display)


    # Histogram Matching Function 
    def perform_histogram_matching(self, target_img, reference_img, display=False):
        
        target_img_path = None
        target_img_name = None
        print(target_img)
        print(reference_img)

        # Checking if the value of the variable is a filepath or an image array 
        # Read Target Image from the path
        if isinstance(target_img, str):
            target_img_path = target_img
            target_img_name = target_img_path.split('/')[-1]
            target_img = skimage.io.imread(target_img)
        # Do nothing if variable is an image
        elif isinstance(target_img, np.ndarray):
            pass
        else:
            raise TypeError("Unkown file type: %{}".format(type(target_img)))
                
        # Checking if the value of the variable is a filepath or an image array 
        # Read Reference Image from the path
        if isinstance(reference_img, str):
            reference_img = skimage.io.imread(reference_img)
        # Do nothing if variable is an image
        elif isinstance(reference_img, np.ndarray):
            pass
        else:
            raise TypeError("Unkown file type: %{}".format(type(reference_img)))

        if len(target_img.shape) != len(reference_img.shape):
            raise ValueError("Target image shape must be the same as the reference image shape") # TODO: is this right?

        # Histogram Equalization to the target image
        print("target_img.shape: ", target_img.shape)
        print("reference_img.shape: ", reference_img.shape)
        if len(target_img.shape) == 3:  
            target_image_equalized = utils.histogram_equalization_3D(target_img)
        else:
            target_image_equalized = utils.histogram_equalization_2D(target_img)

        # Histogram Equalization to the reference image
        if len(reference_img.shape) == 3:    
            reference_image_equalized = utils.histogram_equalization_3D(reference_img)
        else:
            reference_image_equalized = utils.histogram_equalization_2D(reference_img)

        # # Save the histogram equalized target image
        # imageio.imsave('data/test_images/HEtarget.tif', target_img)

        # # Save the histogram equalized reference image
        # imageio.imsave('data/test_images/HErefer.tif', reference_img)

        
        # Find the histogram of the reference image 
        reference_histogram = self.histogram_matcher.get_histogram(reference_image_equalized)

        # Match target image to the reference histogram
        hist_matched_img = self.histogram_matcher.match_image_to_histogram(target_image_equalized, reference_histogram)

        # Result image
        hist_matched_img = np.uint8(hist_matched_img)

        skimage.io.imsave(os.path.join(self.output_path, 'result_image.tif'), hist_matched_img)
        
        if display:
            # Plot
            # Pseudocolor used, which is only relevant to single-channel, grayscale, luminosity images.
            # It can be a useful tool for enhancing contrast and visualizing data more easily
            figure3 = plt.figure()

            #Subplot Layout
            plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)

            # Target Image
            subplot = figure3.add_subplot(321)
            subplot.set_title('Target Image Equalized')
            plt.imshow(np.array(target_image_equalized,np.int32), cmap='gray')

            # Target Image Histogram
            subplot = figure3.add_subplot(322)
            subplot.set_title('Target Image Histogram')
            plt.hist(target_image_equalized.flatten(),256,[0,256])

            # Reference Image
            subplot = figure3.add_subplot(323)
            subplot.set_title('Reference Image Equalized')
            plt.imshow(np.array(reference_image_equalized,np.int32), cmap='gray')

            # Reference Image Histogram
            subplot = figure3.add_subplot(324)
            subplot.set_title('Reference Image Histogram')
            plt.hist(reference_image_equalized.flatten(),256,[0,256])
            
            # Result Image (Matched to Histogram)
            subplot = figure3.add_subplot(325)
            plt.imshow(np.array(hist_matched_img,np.int32), cmap='gray')

            subplot.set_title('Image Matched to Histogram')

            # Save the result histogram matched image
            # imageio.imsave('data/test_images/HMresPNG.png', new_target_img)
            # openHM = skimage.io.imread('data/test_images/HMresPNG.png')
            
            # Result image's Histogram 
            subplot = figure3.add_subplot(326)
            subplot.set_title('Result Image Histogram')
            plt.hist(hist_matched_img.flatten(),256,[0,256])
            plt.xlim([0,256])
            plt.show()
            
            # imageio.imsave('data/test_images/HMres.tif', openHM)