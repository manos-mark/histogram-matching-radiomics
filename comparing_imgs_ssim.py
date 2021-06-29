# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, files, title='Comparison'):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # f1 = files[0].split('TCGA')[2].split('\\')[0]
    # f2 = files[1].split('TCGA')[2].split('\\')[0]
    #     print("--- ---")
    #     print(f"Image 1: {f1} | Image 2: {f2}")
    #     print(f"Mean Square Error: {m} - SSIM: {s}")
    #     print("--- ---")

    # # setup the figure
    # fig = plt.figure(title)
    # plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # # show first image
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(imageA, cmap=plt.cm.gray)
    # plt.axis("off")
    # # show the second image
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(imageB, cmap=plt.cm.gray)
    # plt.axis("off")
    # # show the images
    # plt.show()

    return m, s


def main() -> None:
    results = dict()

    DATASET_PATH = os.path.join('data', 'dataset', 'sygrisampol_images')
    # dirnames = glob.glob(os.path.join(DATASET_PATH, "*", ""))

    post_contrast_imgs = glob.glob(os.path.join(DATASET_PATH, "*_post-contrast.tif"))
    pre_contrast_imgs = glob.glob(os.path.join(DATASET_PATH, "*_pre-contrast.tif"))
    flair_imgs = glob.glob(os.path.join(DATASET_PATH, "*_flair.tif"))

    stuff = [post_contrast_imgs, pre_contrast_imgs, flair_imgs]

    for s in stuff:
        results = dict()
        for i, file in enumerate(s):
            if i == 0:
                img1 = cv2.imread(file)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                f1 = file
            else:
                img2 = cv2.imread(file)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                f2 = file
                files = [f1, f2]
                m, s = compare_images(img1, img2, files)

                results[f1.split('TCGA')[1].split('\\')[0] + ' | ' + f2.split('TCGA')[1].split('\\')[0]] = [m, s]

        print("Final Results Comparisons")
        print("-------------")
        for k, v in results.items():
            print(f'For image : {k}')
            print(f"Mean Square Error: {v[0]}")
            print(f"SSIM: {v[1]}")
            print()
        print("-------------")
        print("-------------")
        print("-------------")
        print("-------------")
        print("-------------")
        print("-------------")

    for s in stuff:
        # Select image with the best contrast
        best_post_constrast_image: dict = {
            'slice': str,
            'contrast_score': 0.0
        }
        for i, file in enumerate(s):
            # Avoid already preprocessed images and masks
            # load images as grayscale
            img = cv2.imread(file, 0)
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

        print(f"The best image is from:"
              f"\nSlice - {best_post_constrast_image['slice']}"
              f"\nContrast Score: {best_post_constrast_image['contrast_score']}")


if __name__ == '__main__':
    main()
