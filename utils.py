import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import imageio
import skimage.io
import nibabel as nib
import glob
import dicom2nifti


def insert_segmenetions_path_to_dict(dataset, new_dataset_output_path, dataset_path, contrast_type):
    for key, value in dataset.items():
        # Get the image path, replace it with the image path from the old dataset
        # and add _roi in order to create the mask path
        path = value['Image'].split('.')                                 # split the path into a list
        path[0] = path[0].replace(new_dataset_output_path, dataset_path) # replace the new path with the old one
        path.insert(1, '_mask.')                                          # append _roi
        path = ''.join(path)
        path = path.replace('_' + contrast_type, '')                                             # join the list elements into a string

        # Add mask path from the old dataset to new dataset dictionary
        dataset[key]['Mask'] = path
        
    return dataset


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
    
    # loop over the slices of the image
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
        img_eq = img_eq.reshape((image.shape[1], image.shape[2]))

        image_equalized[i, :, :] = img_eq

    return image_equalized


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def merge_slices_into_3D_image(dataset_path, contrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))
        
    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.tif"))
        first_dimension = second_dimension = third_dimension = 0
        
        # Count the number of slices and get the shape of the image
        # to initialize the dimensions in order to initialize 
        # the following arrays (mask and image representing the 3D images)
        for file in filenames:
            if "_mask" in file:
                third_dimension += 1
            # Execute this only once
            if first_dimension == 0:
                first_dimension = second_dimension = skimage.io.imread(file).shape[0]
       
        mask = np.zeros([first_dimension, second_dimension, third_dimension], dtype=np.uint8)
        image = np.zeros([first_dimension, second_dimension, third_dimension], dtype=np.uint8)

        for file in filenames:
            i = j = 0
            # Avoid already preprocessed images
            if "_mask" in file:
                mask[:,:,i] = skimage.io.imread(file)
                i += 1
            elif contrast_type in file:
                image[:,:,j] = skimage.io.imread(file)
                j += 1


        image_name = file.rsplit(".")[:-1]
        image_name = '.'.join(image_name)
        image_name = image_name + '_' + contrast_type + '-3D.nii'

        # image = nib.Nifti1Image(image, affine=np.eye(4))
        # nib.save(image, image_name)
        imsave(image_name, image)

        mask_name = file.rsplit(".")[:-1]
        mask_name = '.'.join(mask_name)
        mask_name = mask_name + '_' + contrast_type + '-3D_mask.nii'

        # mask = nib.Nifti1Image(mask, affine=np.eye(4))
        # nib.save(mask, mask_name)
        imsave(mask_name, mask)


def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=True)
    sitk.WriteImage(sitk_img, fname)

    # sitk_img = sitk.GetImageFromArray(np.around(arr*255).astype(np.uint8), isVector=True)
    # sitk.WriteImage(sitk_img, fname)

    # plt.imsave(fname, arr, cmap='gray')

    # plt.imsave(fname, np.around(arr*255).astype(np.uint8), cmap='gray')

    # skimage.io.imsave(fname, arr)

    # skimage.io.imsave(fname, arr, plugin='simpleitk')

    # skimage.io.imsave(fname, np.around(arr*255).astype(np.uint8), plugin='simpleitk')


def split_dataset(dataset_path):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))
        
    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.tif"))

        for file in filenames:
            # Avoid already preprocessed images and masks
            if (not ("_pre-contrast"  in file or "_flair" in file or "_post-contrast" in file or "_mask" in file)):
                img = skimage.io.imread(file)

                filename = file.rsplit(".")[:-1]
                filename = '.'.join(filename)
                
                precontrast_img = filename + '_pre-contrast.tif'
                flair_img = filename + '_flair.tif'
                postcontrast_img = filename + '_post-contrast.tif'

                # Avoid creating again file if exists 
                if not os.path.isfile(precontrast_img):
                    skimage.io.imsave(precontrast_img, img[:,:,0])

                if not os.path.isfile(flair_img):
                    skimage.io.imsave(flair_img, img[:,:,1])

                if not os.path.isfile(postcontrast_img):
                    skimage.io.imsave(postcontrast_img, img[:,:,2])

def get_dataset_as_object(dataset_path, contrast_type):
        cases_dict = {}
        dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

        for dir in dirnames:
            filenames = glob.glob(os.path.join(dir, "*.tif"))

            for file in filenames:

                if "_mask" in file:
                    filename = file.rsplit("_")[:-1]
                    filename = '_'.join(filename)
                    filename = filename.rsplit("/")[2:]
                    filename = ''.join(filename)
                    
                    if filename in cases_dict.keys():
                        cases_dict[filename].update({'Mask': file})
                    else:
                        cases_dict[filename] = {'Mask': file}

                elif file.endswith(contrast_type + ".tif"):
                    filename = file.rsplit(".")[:-1]
                    filename = ''.join(filename)
                    filename = file.rsplit("_")[:-1]
                    filename = '_'.join(filename)
                    filename = filename.rsplit("/")[2:]
                    filename = ''.join(filename)

                    if filename in cases_dict.keys():
                        cases_dict[filename].update({'Image': file})
                    else:
                        cases_dict[filename] = {'Image': file}

            if not cases_dict:
                raise FileNotFoundError("Failed to import dataset.")
            
        return cases_dict


def remove_mask_from_image(img, mask):    

    gray_img = rgb2gray(img)
    gray_mask = rgb2gray(mask)

    # blank = np.zeros(img.shape[:2], dtype='uint8')
    # mask = ~cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

    return cv.bitwise_and(gray_img, gray_mask)
    

def remove_background(img_path):
    dicom2nifti.dicom_series_to_nifti(img_path, "data/dataset/test/test_nifti.nii", reorient_nifti=False)

    # Load a nifti as 3d numpy image [H, W, D]
    nifti = nib.load("data/dataset/test/test_nifti.nii").get_fdata()
    


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