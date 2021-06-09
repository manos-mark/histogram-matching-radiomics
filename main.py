from radiomics_modules.feature_extraction import FeatureExtractor
from histogram_macthing_modules.histogram_matching import HistogramMatcher

import imageio
import utils
import os


def main():

    ######################################################################################################
    ##################################### EXTRACT RADIOMICS FEATURES #####################################

    # Initialize PyRadiomics feature extractor wrapper
    feature_extractor = FeatureExtractor(PARAMETERS_PATH)

    # Import the dataset
    dataset = feature_extractor.import_prepare_dataset(DATASET_PATH)
        
    # Execute batch processing to extract features
    feature_extractor.extract_features(dataset, FEATURES_OUTPUT_PATH)

    # Get the filepaths from the images only (without the segmentations)
    dataset_images = [value['Image'] for value in dataset.values()]


    # ######################################################################################################
    # ######################################### HISTOGRAM MATCHING #########################################

    # # Initialize HistogramMatcher & select histogram matching method
    # histogram_matcher = HistogramMatcher(NEW_DATASET_OUTPUT_PATH, 'ExactHistogramMatching')

    # # # Select Reference Image
    # # reference_img = imageio.imread(os.path.join('data', 'dataset', 'TCGA_CS_4941_19960909', 'TCGA_CS_4941_19960909_10.tif'))
    # # target_img = imageio.imread(os.path.join('data', 'dataset', 'TCGA_CS_4941_19960909', 'TCGA_CS_4941_19960909_11.tif'))
    
    # # histogram_matcher.perform_histogram_matching(target_img, reference_img, display=True)

    # # Perform histogram matching
    # histogram_matcher.perform_batch_histogram_matching(dataset_images, dataset_images[1], display=True) # TODO: dataset[0] is temporal, should we automate reference image selection?


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