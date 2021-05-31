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
import imageio
import scipy
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

        # Checking if the value of the variable is a filepath or an image array 
        # Read Target Image from the path
        if isinstance(target_img, str):
            target_img_path = target_img
            target_img_name = target_img_path.split('/')[-1]
            target_img = imageio.imread(target_img)
        # Do nothing if variable is an image
        elif isinstance(target_img, imageio.core.util.Array):
            pass
        else:
            raise TypeError("Unkown file type: %{}".format(type(target_img)))
        
        # Checking if the value of the variable is a filepath or an image array 
        # Read Reference Image from the path
        if isinstance(reference_img, str):
            reference_img = imageio.imread(reference_img)
        # Do nothing if variable is an image
        elif isinstance(reference_img, imageio.core.util.Array):
            pass
        else:
            raise TypeError("Unkown file type: %{}".format(type(reference_img)))

        if len(target_img.shape) != len(reference_img.shape):
            raise ValueError("Target image shape must be the same as the reference image shape")

        # Histogram Equalization to the target image
        if len(target_img.shape) == 3:  
            target_image_equalized = np.zeros(target_img.shape)
            # loop over the channels of the image
            for i in range(target_img.shape[0]):
                image = target_img[i, :, :]
                target_image_equalized[i, :, :] = self._histogram_equalization(image)[0]
        else:
            target_image_equalized = self._histogram_equalization(target_img)[0]


        # Histogram Equalization to the reference image
        if len(reference_img.shape) == 3:    
            reference_image_equalized = np.zeros(reference_img.shape)
            # loop over the channels of the image
            for i in range(reference_img.shape[0]):
                image = reference_img[i, :, :]
                reference_image_equalized[i, :, :] = self._histogram_equalization(image)[0]
        else:
            reference_image_equalized = self._histogram_equalization(reference_img)[0]

        # # Save the histogram equalized target image
        # imageio.imsave('data/test_images/HEtarget.tif', target_img)

        # # Save the histogram equalized reference image
        # imageio.imsave('data/test_images/HErefer.tif', reference_img)

        
        # Find the histogram of the reference image 
        reference_histogram = self.histogram_matcher.get_histogram(reference_img)

        # Match target image to the reference histogram
        new_target_img = self.histogram_matcher.match_image_to_histogram(target_image_equalized, reference_histogram)
        
        # Result image
        new_target_img = np.uint8(new_target_img)

        if len(target_img.shape) == 2:
            imageio.imsave(os.path.join(self.output_path, 'result_image.tif'), new_target_img)
        else:
            # to save this 3D (ndarry) numpy use this
            func = nib.load(target_img_path)
            ni_img = nib.Nifti1Image(new_target_img, func.affine)
            nib.save(ni_img, os.path.join(self.output_path, target_img_name))
        
        if display:
            # Plot
            # Pseudocolor used, which is only relevant to single-channel, grayscale, luminosity images.
            # It can be a useful tool for enhancing contrast and visualizing data more easily
            figure3 = plt.figure(3)

            #Subplot Layout
            plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)

            # Target Image
            subplot = figure3.add_subplot(221)
            if len(target_img.shape) == 3:
                plt.imshow(np.array(target_img[12,:,:],np.int32))
            else:
                plt.imshow(np.array(target_img,np.int32))
            subplot.set_title('Target Image')

            # Reference Image
            subplot = figure3.add_subplot(222)
            if len(reference_img.shape) == 3:
                plt.imshow(np.array(reference_img[12,:,:],np.int32))
            else:
                plt.imshow(np.array(reference_img,np.int32))
            subplot.set_title('Reference Image')
            
            # Result Image (Matched to Histogram)
            subplot = figure3.add_subplot(223)
            if len(new_target_img.shape) == 3:
                plt.imshow(np.array(new_target_img[12,:,:],np.int32))
            else:
                plt.imshow(np.array(new_target_img,np.int32))

            subplot.set_title('Image Matched to Histogram')

            # Save the result histogram matched image
            # imageio.imsave('data/test_images/HMresPNG.png', new_target_img)
            # openHM = imageio.imread('data/test_images/HMresPNG.png')
            
            # Result image's Histogram 
            subplot = figure3.add_subplot(224)
            subplot.set_title('Result Image Histogram')
            if len(new_target_img.shape) == 3:
                plt.hist(new_target_img[12,:,:].flatten(),256, color = 'b')
            else:
                plt.hist(new_target_img.flatten(),256, color = 'b')
            plt.xlim([0,256])
            plt.show()
            
            # imageio.imsave('data/test_images/HMres.tif', openHM)

    # Histogram Equalization Function
    def _histogram_equalization(self, image, number_bins=256, display=None):
        # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

        # get image histogram
        image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf

    # Histogram Equalization Function
    # Reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    # def _histogram_equalization(self, img, display=None):
    #     cdf = self._getCDF(img);

    #     # The minimum histogram value (excluding 0) by using the Numpy masked array concept
    #     cdf_m = np.ma.masked_equal(cdf,0)
    #     # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
    #     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        
    #     # Look-up table with the information for what is the output pixel value for every input pixel value
    #     cdf = np.ma.filled(cdf_m,0).astype('uint8')

    #     # Apply the transform
    #     imgHE = cdf[img]

    #     if display:
    #         # Plot    
    #         figure2 = plt.figure(2)

    #         # Original Image
    #         subplot2 = figure2.add_subplot(1,2,1)
    #         plt.imshow(np.array(img,np.int32),cmap='gray')
    #         subplot2.set_title('Original Image')

    #         # Histogram Equalized Image
    #         subplot2 = figure2.add_subplot(1,2,2)
    #         plt.imshow(np.array(imgHE,np.int32),cmap='gray')
    #         subplot2.set_title('Histogram Equalized Image')
    #         plt.show()
        
    #     return imgHE


    def _getCDF(self, img, display=None):
        hist, bins = np.histogram(img.flatten(),256)#,[0,256])
        # cdf: Cumulative Distribution Function
        # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
        cdf = hist.cumsum()
        # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_normalized = cdf * hist.max()/ cdf.max()

        # https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy/28520445
        # use linear interpolation of cdf to find new pixel values
        # cdf_normalized = 255 * cdf / cdf[-1] # normalize
        # image_equalized = np.interp(cdf_normalized.flatten(), bins[:-1], cdf)
        # return image_equalized.reshape(image_equalized.shape), cdf

        if display:
            # Plot
            plt.figure(1)

            # Normalized CDF with red
            plt.plot(cdf_normalized, color = 'r')

            # Histogram with black
            plt.hist(img.flatten(),256,[0,256], color = 'k')
            plt.xlim([0,256])

            # Place labels at the lower right of the plot 
            plt.legend(('Normalized CDF','Histogram'), loc = 'lower right')
            plt.show()

        return cdf
    

def main():

    # Initialize HistogramMatcher & Select histogram matching method
    histogram_matcher = HistogramMatcher('data/test_images', 'ExactHistogramMatching')

    # Perform histogram matching
    histogram_matcher.perform_histogram_matching('data/test_images/Fig1.tif', 'data/test_images/Fig2.tif', display=True)


if __name__ == "__main__":
    main()
