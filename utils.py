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
from skimage.color import rgb2gray
import math
import pandas as pd
from skimage.metrics import structural_similarity as ssim

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


def histograms_compare(images,image_names,metric=0):

    image_names = [ i[35:56]  for i in image_names ]
    
    histograms = [ cv.calcHist([x.astype('uint8')], [0], None, [256], [0, 256]) for x in images]
    
    mat = np.zeros((len(images),len(images)))
    
    methods = [cv.HISTCMP_BHATTACHARYYA ]
    
    for i in range(len(images)):
        for j in range(len(images)): 
            mat[i][j]=cv.compareHist(histograms[i],histograms[j],methods[metric])
            
    fig,ax = plt.subplots()
    
    f = np.around(mat,2)
    
    im = ax.imshow(f)

    ax.set_yticks(np.arange(len(image_names)))
    ax.set_yticklabels(image_names)
    
    ax.set_xticks(np.arange(len(image_names)))
    ax.set_xticklabels(image_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    for i in range(len(image_names)):
        for j in range(len(image_names)):
            text = ax.text(j, i, f[i, j], ha="center", va="center", color="w")
            
    val = np.triu(mat).ravel()

    return np.average(len(val[val>0.001]))

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def mse_compare(image,images,image_names,lim=1000,text=''):
    
    image_names = [ i[35:56]  for i in image_names ]
    
    mat = []
   
    for i in range(len(image)):
        mat.append(mse(image[i],images[i]))
        
    plt.figure("mse")
    plot = plt.bar(image_names,mat)
    plt.xticks(rotation=45, ha="right",rotation_mode="anchor")

    return sum(i < lim for i in mat)


def ssim_compare(image,images,image_names,lim=0.5,text=''):
    
    image_names = [ i[35:56]  for i in image_names ]
    
    mat = []
   
    for i in range(len(image)):
        mat.append(ssim(image[i],images[i]))
     
    plt.figure("ssim")
    
    plot = plt.bar(image_names,mat)
    
    plt.xticks(rotation=45, ha="right",rotation_mode="anchor")

    return sum(i > lim for i in mat)

def plot_histograms(images,images_name=''):
    
    image_names = [ i[35:56]  for i in images_name ]
    
    histograms = [ cv.calcHist([x.astype('uint8')], [0], None, [256], [0, 256]) for x in images]

    line =np.arange(0, 256)
    plt.figure(23)
    plt.plot(histograms[1])
    
    plt.figure('original histograms images')
    for i in range(0,len(image_names)):
        plt.xlim(0,255)
        plt.ylim(0, 5000)
        plt.plot(line,histograms[i],label=image_names[i])
        plt.legend(bbox_to_anchor=(.75, 1), borderaxespad=0.)
        plt.show()
    
    images_num = int(math.sqrt(len(image_names)))+1
    
    plt.figure('original images')

    for i in range(0,len(image_names)):
        plt.subplot(images_num,images_num, i+1),plt.imshow(images[i],'gray')
        plt.title(image_names[i])
        plt.show()
   
    return 0


def histogram_equalization_CLAHE(images, number_bins=256, tile_grid_size=(32, 32), clip_limit=2.0,images_name=''):
    
    image_names = [ i[35:56]  for i in images_name ]
    
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    clahe_images = [clahe.apply(x.astype('uint8')) for x in images]
    
    histograms = [ cv.calcHist([x], [0], None, [256], [0, 256]) for x in clahe_images]

    line =np.arange(0, 256)
    
    plt.figure("histograms with clahe clipLimit= "+str(clip_limit)+" tileGridSize ="+ str(tile_grid_size)+'hist')
    for i in range(0,len(image_names)):
        plt.xlim(0,255)
        plt.ylim(0, 5000)
        plt.plot(line,histograms[i],label=image_names[i])
        plt.legend(bbox_to_anchor=(.75, 1), borderaxespad=0.)
        plt.title("histograms with clahe clipLimit= "+str(clip_limit)+" tileGridSize ="+ str(tile_grid_size))
        
        plt.show()
    
    plt.figure("histograms with clahe clipLimit= "+str(clip_limit)+" tileGridSize ="+ str(tile_grid_size)+'img')
    
    images_num =  int(math.sqrt(len(image_names)))+1
    
    for i in range(0,len(image_names)):
        plt.subplot(images_num,images_num, i+1),plt.imshow(clahe_images[i],'gray')
        plt.title(image_names[i])
        plt.show()
   
    return clahe_images

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.
    Code adapted from
    http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    #print(sum(s_counts))
    t_values, t_counts = np.unique(template, return_counts=True)
   # print(s_values)
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)




def histogram_matching(images, ref_img,images_name=''):
    
     histograms = [ cv.calcHist([x.astype('uint8')], [0], None, [256], [0, 256]) for x in images]     
     
     images_name = [ i[35:56]  for i in images_name ]

   
     exact_imgs = [hist_match(i,ref_img) for i in images ]
 
     histograms = [ cv.calcHist([x.astype('uint8')], [0], None, [256], [0, 256]) for x in exact_imgs]

     line =np.arange(0, 256)
     plt.figure('histogram_matching')
     for i in range(0,len(images)):
         plt.xlim(0,255)
         plt.ylim(0, 5000)
         plt.plot(line,histograms[i],label=images_name[i])
         plt.legend(bbox_to_anchor=(.75, 1), borderaxespad=0.)
         plt.show()
        
     images_num = int(math.sqrt(len(images)))+1
     plt.figure('histogram_matching imgs')

     for i in range(0,len(images)):
         plt.subplot(images_num,images_num, i+1),plt.imshow(exact_imgs[i],'gray')
         plt.title(images_name[i])
         plt.show()
 
     return exact_imgs


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


def single_tiff_to_nii(filenames, out_dir, contrast_type, axis=2):
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


def convert_slice_to_nifti(dataset_path, contrast_type):
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
        filenames = glob.glob(os.path.join(dir, "*.png"))

        for file in filenames:
            print(contrast_type + ".tif_removed_background.png")
            if "_mask" in file:
                filename = file.rsplit("_")[:-1]
                filename = '_'.join(filename)
                filename = filename.rsplit("/")[2:]
                filename = ''.join(filename)

                if filename in cases_dict.keys():
                    cases_dict[filename].update({'Mask': file})
                else:
                    cases_dict[filename] = {'Mask': file}

            elif file.endswith(contrast_type + ".tif_removed_background.png"):
                filename = file.rsplit(".")[:-1]
                filename = ''.join(filename)
                filename = file.rsplit("_")[:-1]
                filename = '_'.join(filename)
                filename = filename.rsplit("/")[2:]
                filename = ''.join(filename)
                print(filename)
                if filename in cases_dict.keys():
                    cases_dict[filename].update({'Image': file})
                else:
                    cases_dict[filename] = {'Image': file}

        if not cases_dict:
            raise FileNotFoundError("Failed to import dataset.")

    return cases_dict


def remove_mask_from_image(img, mask, display=False):
    images = np.zeros_like(img)
    masks = np.zeros_like(mask)

    for i in range(img.shape[2]):
        masks[:, :, i] = cv.bitwise_and(img[:, :, i], mask[:, :, i])
        images[:, :, i] = cv.bitwise_xor(img[:, :, i], mask[:, :, i])

        if display:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            ax = axes.ravel()

            ax[0].imshow(img[:, :, i], cmap=plt.cm.gray)
            ax[0].set_title("Original")
            ax[1].imshow(masks[:, :, i], cmap=plt.cm.gray)
            ax[1].set_title("Extracted")

            fig.tight_layout()
            plt.show()

    return images, masks


def add_mask_to_image(img, mask, display=True):
    images = np.zeros_like(img)

    for i in range(img.shape[2]):
        print(img[:, :, i].shape)
        print(mask[:, :, i].shape)
        images[:, :, i] = cv.bitwise_and(img[:, :, i], mask[:, :, i])
        # images[:, :, i] = cv.bitwise_xor(img[:, :, i], mask[:, :, i])

        if display:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            ax = axes.ravel()

            ax[0].imshow(img[:, :, i], cmap=plt.cm.gray)
            ax[0].set_title("Original")
            ax[1].imshow(images[:, :, i], cmap=plt.cm.gray)
            ax[1].set_title("Merged")

            fig.tight_layout()
            plt.show()

    return images


def extract_brain(dataset_path, constrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

    for directory in dirnames:
        filename = glob.glob(os.path.join(directory, '*' + constrast_type + '.nii'))[0].split("/")[-1]

        # Load a nifti as 3d numpy image [H, W, D]
        print("Skull extraction for image: ", os.path.join(filename))
        image = nib.load(os.path.join(directory, filename))
        stripped, mask = robex(image)

        nib.save(stripped, os.path.join(directory, filename))


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

    :param dataset_path:
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
