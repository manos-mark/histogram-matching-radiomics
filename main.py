from radiomics_modules.feature_extraction import FeatureExtractor
from histogram_macthing_modules.histogram_matching import HistogramMatcher

import skimage.io
import utils
import os


def main():

    ######################################################################################################
    ######################################### PREPROCESS DATASET #########################################
    
    # Every Image has 3 channels (pre-contrast, FLAIR, post-contrast) and one mask
    # Import and Split the channels to different folders 
    # It creates new images increasing time and resources consumption
    utils.split_dataset(DATASET_PATH)

    # Prepare dataset for pyradiomics extractor
    # Getting the dataset's path, returns an object specifing for each patient the images and segmentations
    pre_contrast_dataset = utils.get_dataset_as_object(DATASET_PATH, 'pre-contrast')
    flair_dataset = utils.get_dataset_as_object(DATASET_PATH, 'flair')
    post_contrast_dataset = utils.get_dataset_as_object(DATASET_PATH, 'post-contrast')


    ######################################################################################################
    ##################################### EXTRACT RADIOMICS FEATURES #####################################

    # Initialize PyRadiomics feature extractor wrapper
    feature_extractor = FeatureExtractor(PARAMETERS_PATH)
        
    # Execute batch processing to extract features
    feature_extractor.extract_features(pre_contrast_dataset, FEATURES_OUTPUT_PATH)

    # Get the filepaths from the images only (without the segmentations)
    dataset_images = [value['Image'] for value in pre_contrast_dataset.values()]


    # ######################################################################################################
    # ######################################### HISTOGRAM MATCHING #########################################

    # Initialize HistogramMatcher & select histogram matching method
    histogram_matcher = HistogramMatcher(NEW_DATASET_OUTPUT_PATH, 'ExactHistogramMatching')

    # Select Reference Image
    # reference_img = skimage.io.imread(os.path.join('data', 'dataset', 'TCGA_CS_4941_19960909', 'TCGA_CS_4941_19960909_10.tif'))
    # target_img = skimage.io.imread(os.path.join('data', 'dataset', 'TCGA_CS_4941_19960909', 'TCGA_CS_4941_19960909_11.tif'))
    print(pre_contrast_dataset[0])
    histogram_matcher.perform_histogram_matching(pre_contrast_dataset[0], pre_contrast_dataset[1], display=True)

    # # Perform histogram matching
    # histogram_matcher.perform_batch_histogram_matching(pre_contrast_dataset, pre_contrast_dataset[1], display=True) # TODO: dataset[0] is temporal, should we automate reference image selection?


    # ######################################################################################################
    # ##################################### EXTRACT RADIOMICS FEATURES #####################################
    # #####################################    FROM THE NEW DATASET    #####################################

    # # Import the dataset
    # new_dataset = feature_extractor.import_prepare_dataset(NEW_DATASET_OUTPUT_PATH)

    # # We don't have the segmentations on the new dataset folder because we created it in the previous step
    # # by applying histogram matching on the image, not the segmentation.
    # # So we need to copy the segmentations paths from the old dataset and add them to the new dataset's dictionary
    # new_dataset = utils.insert_segmenetions_path_to_dict(new_dataset, NEW_DATASET_OUTPUT_PATH, DATASET_PATH)

    # # Execute batch processing to extract features
    # feature_extractor.extract_features(new_dataset, NEW_FEATURES_OUTPUT_PATH)

    # # Get the filepaths from the images only (without the segmentations)
    # new_dataset = [value['Image'] for value in new_dataset.values()]

    
    # ######################################################################################################
    # ########################################### COMPARE RESULTS ##########################################

    

if __name__ == '__main__':

    PARAMETERS_PATH = os.path.join('radiomics_modules', 'Params.yaml')

    DATASET_PATH = os.path.join('data', 'dataset')
    FEATURES_OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')

    NEW_DATASET_OUTPUT_PATH = os.path.join('data', 'new_dataset')
    NEW_FEATURES_OUTPUT_PATH = os.path.join('data', 'new_pyradiomics_extracted_features.csv')

    main()