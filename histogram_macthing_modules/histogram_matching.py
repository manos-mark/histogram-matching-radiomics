# Exact Histogram Specification and Histogram Equalization
# Tsichlaki Styliani mtp209
# Advances in Digital Imaging and Computer Vision
# env Python 3.7
# 1st Implementation

# libs and imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import matplotlib.image as mpimg
import numpy as np
import skimage
import scipy
import cv2
# Import Stefano Di Martino's function
# Reference: https://github.com/StefanoD/ExactHistogramSpecification
from exact_histogram_matching import ExactHistogramMatcher

# Histogram Matching Function 
def histogram_matching(original_img, specified_img):
    # Target image
    target_img = original_img
    # Reference Image
    reference_img = specified_img

    # Histogram Equalization to the target image
    target_img = histogram_equalization(target_img)
    # Histogram Equalization to the reference image
    reference_img = histogram_equalization(reference_img)

    # Save the histogram equalized target image
    imageio.imsave('data/test_images/HEtarget.tif', target_img)
    # Save the histogram equalized reference image
    imageio.imsave('data/test_images/HErefer.tif', reference_img)

    
    # Find the histogram of the reference image 
    reference_histogram = ExactHistogramMatcher.get_histogram(reference_img)
    # Match target image to the reference histogram
    new_target_img = ExactHistogramMatcher.match_image_to_histogram(target_img, reference_histogram)
    # Result image
    #res = np.uint8(new_target_img)
    
    # Plot
    # Pseudocolor used, which is only relevant to single-channel, grayscale, luminosity images.
    # It can be a useful tool for enhancing contrast and visualizing data more easily
    figure3 = plt.figure(3)

    #Subplot Layout
    plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)

    # Target Image
    subplot = figure3.add_subplot(221)
    plt.imshow(np.array(target_img,np.int32))
    subplot.set_title('Target Image')

    # Reference Image
    subplot = figure3.add_subplot(222)
    plt.imshow(np.array(reference_img,np.int32))
    subplot.set_title('Reference Image')
    
    # Result Image (Matched to Histogram)
    subplot = figure3.add_subplot(223)
    plt.imshow(np.array(new_target_img,np.int32))
    subplot.set_title('Image Matched to Histogram')

    # Save the result histogram matched image
    imageio.imsave('data/test_images/HMresPNG.png', new_target_img);
    openHM = imageio.imread('data/test_images/HMresPNG.png')
    
    # Result image's Histogram 
    subplot = figure3.add_subplot(224)
    subplot.set_title('Result Image Histogram')
    plt.hist(openHM.flatten(),256,[0,256], color = 'b')
    plt.xlim([0,256])
    plt.show()
    imageio.imsave('data/test_images/HMres.tif', openHM);

# Histogram Equalization Function
# Reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histogram_equalization(img):
    cdf = getCDF(img);

    # The minimum histogram value (excluding 0) by using the Numpy masked array concept
    cdf_m = np.ma.masked_equal(cdf,0)
    # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    
    # Look-up table with the information for what is the output pixel value for every input pixel value
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    # Apply the transform
    imgHE = cdf[img]

    # Plot    
    figure2 = plt.figure(2)

    # Original Image
    subplot2 = figure2.add_subplot(1,2,1)
    plt.imshow(np.array(img,np.int32),cmap='gray')
    subplot2.set_title('Original Image')

    # Histogram Equalized Image
    subplot2 = figure2.add_subplot(1,2,2)
    plt.imshow(np.array(imgHE,np.int32),cmap='gray')
    subplot2.set_title('Histogram Equalized Image')
    plt.show()
    
    return imgHE

def getCDF(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    # cdf: Cumulative Distribution Function
    # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
    cdf = hist.cumsum()
    # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_normalized = cdf * hist.max()/ cdf.max()

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
    # Target Image
    original_img = imageio.imread('data/test_images/Fig1.tif')
    # Reference Image
    specified_img = imageio.imread('data/test_images/Fig2.tif')
    histogram_matching(original_img, specified_img)

if __name__ == "__main__":
    main()
