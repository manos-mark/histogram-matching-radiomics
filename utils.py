import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def remove_mask_from_image(img, mask):    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    # blank = np.zeros(img.shape[:2], dtype='uint8')
    # mask = ~cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

    masked = cv.bitwise_and(gray, mask)

    cv.imshow('Masked Image', masked)

    """ Graylevel Histogram """
    gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256])

    plt.figure()
    plt.title('Graylevel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])
    plt.show()

    cv.waitKey(0)

    return masked


if __name__ == "__main__":
    image_path = 'data/test_images/R01-001.nii'
    mask_path = 'data/test_images/R01-001_roi.nii'
    
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)[12,:,:]

    mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask)[12,:,:]

    plt.figure(figsize=(20,20))

    plt.subplot(2,2,1)
    plt.imshow(image, cmap="gray")
    plt.title("Brain")

    plt.subplot(2,2,2)
    plt.imshow(mask)        
    plt.title("Segmentation")

    masked_image = remove_mask_from_image(image, mask)

    # plt.subplot(2,2,3)
    # plt.imshow(sitk.GetArrayFromImage(masked_image)[12,:,:])        
    # plt.title("Masked Image")

    plt.show()