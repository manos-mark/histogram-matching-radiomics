from radiomics_modules.feature_extraction import FeatureExtractor
from histogram_macthing_modules.histogram_matching import HistogramMatcher

import skimage.io
import cv2 as cv
import utils
import os
import glob

def main():
    ######################################################################################################
  
    #====  experiments ===========================================================================
    
    folders = os.listdir(DATASET_PATH)
    
    #onomata arxikon eikonon 
    
    pos_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*post-contrast.tif_removed_background.png"))[0] for folder in folders]

    flair_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*flair.tif_removed_background.png"))[0] for folder in folders]
    
    pre_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*pre-contrast.tif_removed_background.png"))[0] for folder in folders]
    
    all_images = [pos_images_names, flair_images_names,pre_images_names]
    

    
    channel = 0 #epilogi kanalioy
    
    images_names = all_images[channel] 
    
    images = [ cv.imread(x, 0)  for x in images_names ]
    
#    #ektiposi arxikon instogrammaton 
#    
    #utils.plot_histograms(images,images_names)
#    
#    #sigrisi arxikon eikonon
#    
#    print(utils.histograms_compare(images,images_names,metric=0))
    
#
#CLAHE ===========================================================================================================
#    
#    clahe_images =utils.histogram_equalization_CLAHE(images,tile_grid_size=(24,24), clip_limit=5,images_name=images_names)
#    
#    print(utils.histograms_compare(clahe_images,images_names,metric=0))
#    
#    print(utils.ssim_compare(clahe_images,images,images_names))
#    
#    print(utils.mse_compare(clahe_images,images,images_names))
   
#histogram matching==========================================================================================
 
#    ref_image = ['data/dataset/TCGA_FG_5964_20010511/TCGA_FG_5964_20010511_5_post-contrast.tif_removed_background.png',
#                 'data/dataset/TCGA_FG_6692_20020606/TCGA_FG_6692_20020606_4_flair.tif_removed_background.png',
#                 'data/dataset/TCGA_FG_A4MT_20020212/TCGA_FG_A4MT_20020212_5_pre-contrast.tif_removed_background.png']
#    
#    utils.plot_histograms(images,images_names)
#
#    hist_images = utils.histogram_matching(images,cv.imread(ref_image[channel], 0) ,images_name=images_names)
#    
#    print(utils.histograms_compare(hist_images,images_names,metric=0))
#    print(utils.ssim_compare(hist_images,images,images_names))
#    
#    print(utils.mse_compare(hist_images,images,images_names))
    

#pipeline =========================================================================================================
    
    ref_image = ['data/dataset/TCGA_FG_5964_20010511/TCGA_FG_5964_20010511_5_post-contrast.tif_removed_background.png',
                 'data/dataset/TCGA_FG_6692_20020606/TCGA_FG_6692_20020606_4_flair.tif_removed_background.png',
                 'data/dataset/TCGA_FG_A4MT_20020212/TCGA_FG_A4MT_20020212_5_pre-contrast.tif_removed_background.png']
    
                 
    hist_images = utils.histogram_matching(images,cv.imread(ref_image[channel], 0) ,images_name=images_names)
    
    final_images = utils.histogram_equalization_CLAHE(hist_images,tile_grid_size=(24,24), clip_limit=10,images_name=images_names)
  
    print(utils.histograms_compare(final_images,images_names,metric=0))
    
    print(utils.ssim_compare(final_images,images,images_names))
    
    print(utils.mse_compare(final_images,images,images_names))
    

  
#    text = '/CLAHE_CLIP_5'
#    for i in range(0,len(images)):
#        if not os.path.exists('images/'+pos_images_names[i][-64:-27]):
#            os.makedirs('images/'+pos_images_names[i][-64:-27])
#            cv.imwrite('images/'+pos_images_names[i][-64:-27]+text+'.png',images[i]) 

    ######################################################################################################
    ##################################### EXTRACT RADIOMICS FEATURES #####################################
    #####################################    FROM THE NEW DATASET    #####################################

    # Import the dataset
    # new_dataset = utils.get_dataset_as_object(NEW_DATASET_OUTPUT_PATH, 'flair')

    # We don't have the segmentations on the new dataset folder because we created it in the previous step
    # by applying histogram matching on the image, not the segmentation.
    # So we need to copy the segmentations paths from the old dataset and add them to the new dataset's dictionary
    # new_dataset = utils.insert_segmenetions_path_to_dict(new_dataset, NEW_DATASET_OUTPUT_PATH, DATASET_PATH, 'flair')

    # Execute batch processing to extract features
    # feature_extractor.extract_features(new_dataset, NEW_FEATURES_OUTPUT_PATH)

    # Get the filepaths from the images only (without the segmentations)
    # new_dataset = [value['Image'] for value in new_dataset.values()]

    ######################################################################################################
    ########################################### COMPARE RESULTS ##########################################


if __name__ == '__main__':
    PARAMETERS_PATH = os.path.join('radiomics_modules', 'Params.yaml')

    DATASET_PATH = os.path.join('data', 'dataset')
    FEATURES_OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')

    NEW_DATASET_OUTPUT_PATH = os.path.join('data', 'new_dataset')
    NEW_FEATURES_OUTPUT_PATH = os.path.join('data', 'new_pyradiomics_extracted_features.csv')

    main()
