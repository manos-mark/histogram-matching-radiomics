import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import imageio
import skimage.io
import napari
import nibabel as nib
from pyrobex.robex import robex
import glob
import dicom2nifti
from PIL import Image


def insert_segmenetions_path_to_dict(dataset, new_dataset_output_path, dataset_path, contrast_type):
    for key, value in dataset.items():
        # Get the image path, replace it with the image path from the old dataset
        # and add _roi in order to create the mask path
        path = value['Image'].split('.')  # split the path into a list
        path[0] = path[0].replace(new_dataset_output_path, dataset_path)  # replace the new path with the old one
        path.insert(1, '_mask.')  # append _roi
        path = ''.join(path)
        path = path.replace('_' + contrast_type, '')  # join the list elements into a string

        # Add mask path from the old dataset to new dataset dictionary
        dataset[key]['Mask'] = path

    return dataset


def histogram_equalization_CLAHE(img, number_bins=256, tile_grid_size=(32, 32), clip_limit=2.0):
    print(img)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    image = cv.resize(img, (200, 200), interpolation=cv.INTER_AREA)

    clahe_image = clahe.apply(image)

    # clahe_histograms = [cv.calcHist([x], [0], None, [256], [0, 256]) for x in clahe_images]

    return clahe_image


# Histogram Equalization Function
# Reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histogram_equalization_2D(img, number_bins=256, display=False):
    # cdf, bins = getCDF(img, display)
    hist, bins = np.histogram(img.flatten(), number_bins, [0, 256])
    # cdf: Cumulative Distribution Function
    # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
    cdf = hist.cumsum()
    # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_normalized = cdf * hist.max() / cdf.max()

    # The minimum histogram value (excluding 0) by using the Numpy masked array concept
    cdf_m = np.ma.masked_equal(cdf_normalized, 0)
    # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # Look-up table with the information for what is the output pixel value for every input pixel value
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply the transform
    image_equalized = cdf[img]

    if display:
        # Plot    
        figure2 = plt.figure(2)

        # Original Image
        subplot2 = figure2.add_subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        subplot2.set_title('Original Image')

        # Histogram Equalized Image
        subplot2 = figure2.add_subplot(1, 2, 2)
        plt.imshow(image_equalized, cmap='gray')
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
        hist, bins = np.histogram(img.flatten(), number_bins, density=True)  # , [0,256])
        cdf = hist.cumsum()  # cumulative distribution function
        cdf = cdf * hist.max() / cdf.max()  # 255 * cdf / cdf[-1] # normalize

        # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_normalized = cdf * hist.max() / cdf.max()

        # The minimum histogram value (excluding 0) by using the Numpy masked array concept
        cdf_m = np.ma.masked_equal(cdf_normalized, 0)
        # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

        # Look-up table with the information for what is the output pixel value for every input pixel value
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        # https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy/28520445
        # use linear interpolation of cdf to find new pixel values (for 3D images)
        img_eq = np.interp(img.flatten(), bins[:-1], cdf)
        img_eq = img_eq.reshape((image.shape[1], image.shape[2]))

        image_equalized[i, :, :] = img_eq

    return image_equalized


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def single_tiff_to_nii(filenames,out_dir,contrast_type,axis=2) :
    try:
        correct_headers_path = os.path.join('data', 'correct_headers')
        correct_headers_nifti_path = os.path.join(correct_headers_path, 'img_with_correct_header.nii')

        dicom2nifti.dicom_series_to_nifti(correct_headers_path, correct_headers_nifti_path, reorient_nifti=False)

        img_with_correct_header = nib.load(correct_headers_nifti_path)
        affine = img_with_correct_header.affine
        header = img_with_correct_header.header

        fns = sorted(filenames)
        for fn in fns:
            imgs = []
            _, base, ext = split_filename(fn)
            if 'mask' in base:
                base = base.split('_')[:-1]
                base.append('mask')
                base = '_'.join(base)

            else:
                base = base.split('_')[:-1]
                base.append(contrast_type)
                base = '_'.join(base)

            img = np.asarray(Image.open(fn)).astype(np.float32).squeeze()
            img = np.rot90(img)

            if img.ndim != 2:
                raise Exception(f'Only 2D data supported. File {base}{ext} has dimension {img.ndim}.')
            imgs.append(img)
            img = np.stack(imgs, axis=axis)
            nib.Nifti1Image(img, affine, header).to_filename(os.path.join(out_dir, f'{base}_single.nii'))
        return 0
    except Exception as e:
        print(e)
        return 1


def tiff_to_nii(images, out_dir, contrast_type, axis=2):
    try:
        correct_headers_path = os.path.join('data', 'correct_headers')
        correct_headers_nifti_path = os.path.join(correct_headers_path, 'img_with_correct_header.nii')

        dicom2nifti.dicom_series_to_nifti(correct_headers_path, correct_headers_nifti_path, reorient_nifti=False)

        img_with_correct_header = nib.load(correct_headers_nifti_path)
        affine = img_with_correct_header.affine
        header = img_with_correct_header.header

        fns = sorted(images)
        imgs = []
        for fn in fns:
            _, base, ext = split_filename(fn)
            if 'mask' in base:
                base = base.split('_')[:-2]
                base.append('mask')
                base = '_'.join(base)

            else:
                base = base.split('_')[:-2]
                base.append(contrast_type)
                base = '_'.join(base)

            img = np.asarray(Image.open(fn)).astype(np.float32).squeeze()
            img = np.rot90(img)

            if img.ndim != 2:
                raise Exception(f'Only 2D data supported. File {base}{ext} has dimension {img.ndim}.')
            imgs.append(img)
        img = np.stack(imgs, axis=axis)
        nib.Nifti1Image(img, affine, header).to_filename(os.path.join(out_dir, f'{base}.nii'))
        return 0
    except Exception as e:
        print(e)
        return 1


def merge_slices_into_3D_image(dataset_path, contrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, '*_' + contrast_type + '.tif'))
        masknames = glob.glob(os.path.join(dir, "*_mask.tif"))

        if not filenames or not masknames:
            raise ValueError(f'dir ({dir}) does not contain any .tif or .tiff images.')

        tiff_to_nii(filenames, dir, contrast_type)
        tiff_to_nii(masknames, dir, contrast_type)

def getSingleImageNii(dataset_path, contrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, '*_' + contrast_type + '.tif'))

        if not filenames:
            raise ValueError(f'dir ({dir}) does not contain any .tif or .tiff images.')

        single_tiff_to_nii(filenames, dir, contrast_type)


def split_dataset(dataset_path):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.tif"))

        for file in filenames:
            # Avoid already preprocessed images and masks
            if not ("_pre-contrast" in file or "_flair" in file or "_post-contrast" in file or "_mask" in file):
                img = skimage.io.imread(file)

                filename = file.rsplit(".")[:-1]
                filename = '.'.join(filename)

                precontrast_img = filename + '_pre-contrast.tif'
                flair_img = filename + '_flair.tif'
                postcontrast_img = filename + '_post-contrast.tif'

                # Avoid creating again file if exists 
                if not os.path.isfile(precontrast_img):
                    skimage.io.imsave(precontrast_img, img[:, :, 0])

                if not os.path.isfile(flair_img):
                    skimage.io.imsave(flair_img, img[:, :, 1])

                if not os.path.isfile(postcontrast_img):
                    skimage.io.imsave(postcontrast_img, img[:, :, 2])


def get_dataset_as_object(dataset_path, contrast_type):
    cases_dict = {}
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.nii"))

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

            elif file.endswith(contrast_type + ".nii"):
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


def remove_background(dataset_path, constrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for dir in dirnames:
        filename = glob.glob(os.path.join(dir, '*' + constrast_type + '.nii'))[0].split("/")[-1]

        # Load a nifti as 3d numpy image [H, W, D]
        print("Skull extraction for image: ", os.path.join(filename))
        image = nib.load(os.path.join(filename))
        stripped, mask = robex(image)

        nib.save(stripped, os.path.join(dir, filename))


def convert_images_to_3d_numpy_arrays(base_path: str, mode: str, directories: list) -> dict:
    """
    Concatenates multiple 2D images to a 3D one
    :param base_path: Base path for each patient
    :param mode: .tif mode that you want to produce a 3D of
    :param directories: Edw mpainei to stuff
    :return: Returns a dictionary that has 'Key:Value' pairs of 'Name_Of_Folder:3D_Image'
        3D Image is essentially a Numpy array
    """
    try:
        # Concatenate every _pre-contrast.tif file
        file_extension: str = f'*_{mode}.tif'
        # Every 3D Image (numpy array) in a dictionary
        images_in_3d = dict()

        # Create a 3D Image from each folder/patient
        for patient in directories:
            # Define Dataset Path
            images_path = os.path.join(base_path, patient, file_extension)
            # Read all images from the path as an ImageCollection
            im_collection: skimage.io.ImageCollection = skimage.io.ImageCollection(images_path)
            # Each Value in the dictionary is a file
            images_in_3d[patient] = im_collection.concatenate()

        return images_in_3d

    except Exception as e:
        print(e.__str__())


def display_3d_images(images: dict) -> None:
    """
    --- Runs properly only with Jupyter Notebook ---
    --- napari library also requires pyqt5 library, which you install separately ---

    Display the image for reference
    It now prints only one image, for reference

    :param images:
    :return: None
    """
    try:
        one_image = images['TCGA_CS_4941_19960909']
        viewer = napari.view_image(one_image)
        napari.run()

    except Exception as e:
        print(e.__str__())


def evaluate_processed_images(dataset_path) -> None:
    """
    Rates the contrast of images

    :param stuff:
    :return:
    """
    dirnames: str = glob.glob(os.path.join(dataset_path, "*", ""))
    best_post_constrast_image: dict = {
        'slice': str,
        'contrast_score': 0.0
    }

    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.tif"))

        for file in filenames:
            # Avoid already preprocessed images and masks
            if "_post-contrast" in file:
                # load images as grayscale
                img = cv.imread(file, 0)
                # img = cv2.imread("TCGA_CS_4941_19960909_11_post-contrast.tif", 0)
                hh, ww = img.shape[:2]

                # compute total pixels
                tot = hh * ww

                # compute histogram
                hist = np.histogram(img, bins=256, range=[0, 255])[0]

                # compute cumulative histogram
                cum = np.cumsum(hist)

                # normalize histogram to range 0 to 100
                cum = 100 * cum / tot

                # get bins of percentile at 25 and 75 percent in cum histogram
                i = 0
                while cum[i] < 25:
                    i = i + 1
                B1 = i
                i = 0
                while cum[i] < 75:
                    i = i + 1
                B3 = i
                # print('25 and 75 percentile bins:', B1, B3)

                # compute min and max graylevel (which are also the min and max bins)
                min = np.amin(img)
                max = np.amax(img)
                # print('min:', min, 'max:', max)

                # compute contrast
                contrast = (B3 - B1) / (max - min)
                # print('contrast:', contrast)

                if contrast > best_post_constrast_image.get('contrast_score'):
                    best_post_constrast_image['slice'] = file
                    best_post_constrast_image['contrast_score'] = contrast

    print(f"The best post-contrast image is from:"
          f"\nSlice - {best_post_constrast_image['slice']}"
          f"\nContrast Score: {best_post_constrast_image['contrast_score']}")


if __name__ == "__main__":
    image_path = 'data/dataset/R01-001.nii'
    mask_path = 'data/dataset/R01-001_roi.nii'

    # image = sitk.ReadImage(image_path)
    # image = sitk.GetArrayFromImage(image)

    # mask = sitk.ReadImage(mask_path)
    # mask = sitk.GetArrayFromImage(mask)
    image = imageio.imread(image_path)

    mask = imageio.imread(mask_path)

    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    plt.imshow(image[12, :, :], cmap="gray")
    plt.title("Brain")

    plt.subplot(2, 2, 2)
    plt.imshow(mask[12, :, :], cmap="gray")
    plt.title("Segmentation")

    masked_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        img = image[i, :, :]
        msk = mask[i, :, :]
        masked_image[i, :, :] = remove_mask_from_image(img, msk)

    plt.subplot(2, 2, 3)
    plt.imshow(masked_image[12, :, :], cmap='gray')
    plt.title("Masked Image")

    plt.subplot(2, 2, 4)
    plt.title('Graylevel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixels')
    plt.hist(masked_image[12, :, :].flatten(), 256, [0, 256], color='b')
    # plt.hist(np.histogram(masked_image.flatten(),256))
    plt.xlim([0, 256])

    plt.show()

    cv.waitKey(0)
