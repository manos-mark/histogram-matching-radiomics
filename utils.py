import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import imageio


# Histogram Equalization Function
# Reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histogram_equalization_2D(img, number_bins=256, display=False):
    # cdf, bins = getCDF(img, display)
    hist, bins = np.histogram(img.flatten(), number_bins, [0,256])
    # cdf: Cumulative Distribution Function
    # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
    cdf = hist.cumsum()
    # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_normalized = cdf * hist.max()/ cdf.max()

    # The minimum histogram value (excluding 0) by using the Numpy masked array concept
    cdf_m = np.ma.masked_equal(cdf_normalized,0)
    # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    
    # Look-up table with the information for what is the output pixel value for every input pixel value
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    # Apply the transform
    image_equalized = cdf[img]

    if display:
        # Plot    
        figure2 = plt.figure(2)

        # Original Image
        subplot2 = figure2.add_subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        subplot2.set_title('Original Image')

        # Histogram Equalized Image
        subplot2 = figure2.add_subplot(1,2,2)
        plt.imshow(image_equalized ,cmap='gray')
        subplot2.set_title('Histogram Equalized Image')
        plt.show()

    return image_equalized

    
# Histogram Equalization Function
def histogram_equalization_3D(image, number_bins=256):
    image_equalized = np.zeros(image.shape)
    
    # loop over the channels of the image
    for i in range(image.shape[0]):
        img = image[i, :, :]

        # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
        # get image histogram
        hist, bins = np.histogram(img.flatten(), number_bins)#, [0,256])
        cdf = hist.cumsum() # cumulative distribution function
        cdf = cdf * hist.max()/ cdf.max()#255 * cdf / cdf[-1] # normalize

        # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_normalized = cdf * hist.max()/ cdf.max()

        # The minimum histogram value (excluding 0) by using the Numpy masked array concept
        cdf_m = np.ma.masked_equal(cdf_normalized,0)
        # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        
        # Look-up table with the information for what is the output pixel value for every input pixel value
        cdf = np.ma.filled(cdf_m,0).astype('uint8')

        # https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy/28520445
        # use linear interpolation of cdf to find new pixel values (for 3D images)
        img_eq = np.interp(img.flatten(), bins[:-1], cdf)
        img_eq = img_eq.reshape(img.shape)

        image_equalized[i, :, :] = img_eq

    return image_equalized


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def remove_mask_from_image(img, mask):    

    gray_img = rgb2gray(img)
    gray_mask = rgb2gray(mask)

    # blank = np.zeros(img.shape[:2], dtype='uint8')
    # mask = ~cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

    return cv.bitwise_and(gray_img, gray_mask)
    

def getCDF(img, display=None):
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    # cdf: Cumulative Distribution Function
    # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
    cdf = hist.cumsum()
    # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_normalized = cdf * hist.max()/ cdf.max()

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

    return cdf, bins


if __name__ == "__main__":
    image_path = 'data/dataset/R01-001.nii'
    mask_path = 'data/dataset/R01-001_roi.nii'
    
    # image = sitk.ReadImage(image_path)
    # image = sitk.GetArrayFromImage(image)
    
    # mask = sitk.ReadImage(mask_path)
    # mask = sitk.GetArrayFromImage(mask)
    image = imageio.imread(image_path)
    
    mask = imageio.imread(mask_path)

    plt.figure(figsize=(20,20))

    plt.subplot(2,2,1)
    plt.imshow(image[12,:,:], cmap="gray")
    plt.title("Brain")

    plt.subplot(2,2,2)
    plt.imshow(mask[12,:,:], cmap="gray")       
    plt.title("Segmentation")

    masked_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        img = image[i, :, :]
        msk = mask[i, :, :]
        masked_image[i, :, :] = remove_mask_from_image(img, msk)


    plt.subplot(2,2,3)
    plt.imshow(masked_image[12,:,:], cmap='gray')        
    plt.title("Masked Image")

    plt.subplot(2,2,4)
    plt.title('Graylevel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixels')
    plt.hist(masked_image[12,:,:].flatten(), 256,[0,256], color = 'b')
    # plt.hist(np.histogram(masked_image.flatten(),256))
    plt.xlim([0,256])

    plt.show()

    cv.waitKey(0)