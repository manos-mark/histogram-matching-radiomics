from radiomics_modules.feature_extraction import FeatureExtractor
from histogram_macthing_modules.histogram_matching import HistogramMatcher

import skimage.io
import cv2 as cv
import utils
import os
import glob

def main():
    ######################################################################################################
    ######################################### PREPROCESS DATASET #########################################

    # Every Image has 3 channels (pre-contrast, FLAIR, post-contrast) and one mask
    # Import and Split the channels to different folders 
    # It creates new images increasing time and resources consumption
    utils.split_dataset(DATASET_PATH)

    # Disable this for faster Debugging
    utils.merge_slices_into_3D_image(DATASET_PATH, 'post-contrast')

    # SINGLE POST CONTRAST IMAGE NIFTI - EXTRACTION
    utils.convert_slice_to_nifti(DATASET_PATH, 'post-contrast')

    # Disable this for faster Debugging
    # utils.extract_brain(DATASET_PATH, 'post-contrast')

    # Prepare dataset for pyradiomics extractor
    # Getting the dataset's path, returns an object specifing for each patient the images and segmentations
    # pre_contrast_dataset = utils.get_dataset_as_object(DATASET_PATH, 'pre-contrast')  # pre-contrast-3D
    # flair_dataset = utils.get_dataset_as_object(DATASET_PATH, 'flair')
    post_contrast_dataset = utils.get_dataset_as_object(DATASET_PATH, 'post-contrast')

    # Get the filepaths from the images only (without the segmentations) as a list
    # pre_contrast_images = [value['Image'] for value in pre_contrast_dataset.values()]
    # flair_images = [value['Image'] for value in flair_dataset.values()]
    post_contrast_images = [value['Image'] for value in post_contrast_dataset.values()]

    ######################################################################################################
    ##################################### EXTRACT RADIOMICS FEATURES #####################################

    # Initialize PyRadiomics feature extractor wrapper
    feature_extractor = FeatureExtractor(PARAMETERS_PATH)

    # Execute batch processing to extract features
    feature_extractor.extract_features(post_contrast_dataset, FEATURES_OUTPUT_PATH)

    # ######################################################################################################
    # ######################################### HISTOGRAM MATCHING #########################################

    # Initialize HistogramMatcher & select histogram matching method
    # histogram_matcher = HistogramMatcher(NEW_DATASET_OUTPUT_PATH, 'ExactHistogramMatching')

    # Perform histogram matching
    # histogram_matcher.match_histograms(flair_images[0], flair_images[1], display=True)

    # Perform Batch histogram matching
    # histogram_matcher.match_histograms(post_contrast_images, post_contrast_images[0], using_mask_extraction=True, display=True) # TODO: dataset[0] is temporal, should we automate reference image selection?
    #exact histogram matching
    
    #====  NEW ===========================================================================


    folders = os.listdir(DATASET_PATH)
    print(DATASET_PATH)
    
    #onomata arxikon eikonon 
    
    pos_images_names = [glob.glob(os.path.join(DATASET_PATH, folder, "*post-contrast.tif_removed_background.png"))[0] for folder in folders]
    #diavasma arxikon eikonon 
    
    original_images = [ cv.imread(x, 0)  for x in pos_images_names]
    
    #eikones me clahe 
    clahe_images =utils.histogram_equalization_CLAHE(original_images,tile_grid_size=(24,24), clip_limit=5.0,images_name=pos_images_names)
    
    #exact histogram matching
    
    ref_image = 'data/dataset/TCGA_FG_5964_20010511/TCGA_FG_5964_20010511_5_post-contrast.tif_removed_background.png'
 
    exact_hist_images = utils.exact_histogram_matching(original_images,cv.imread(ref_image, 0) ,images_name=pos_images_names)
    
    
    utils.histograms_compare(exact_hist_images,pos_images_names,metric=0)
    
    utils.ssim_compare(exact_hist_images,original_images,pos_images_names)
    
    
    
    
    #save clahe images
 #   for j in range(0,len(clip_lim)):
  #      for i in range(0,len(pos_images_names)):
  #          if not os.path.exists('images/'+pos_images_names[i][-64:-27]):
  #              os.makedirs('images/'+pos_images_names[i][-64:-27])
  #          cv.imwrite('images/'+pos_images_names[i][-64:-27]+'/CLAHE_CLIP_LIM_'+str(clip_lim[j])+'.png',images[j][i]) 
        
    #save exact hist match
    
 #   for i in range(0,len(pos_images_names)):
   #             cv.imwrite('images/'+pos_images_names[i][-64:-27]+'/EXACT_HIST.png',exact_matching_final[i]) 

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
