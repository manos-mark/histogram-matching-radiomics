from radiomics_modules.feature_extraction import FeatureExtractor
from histogram_macthing_modules.histogram_matching import HistogramMatcher

import imageio
import os


def main():

    ######################################################################################################
    ##################################### EXTRACT RADIOMICS FEATURES #####################################

    # Initialize PyRadiomics feature extractor wrapper
    feature_extractor = FeatureExtractor(PARAMETERS_PATH)

    # Import the dataset
    dataset = feature_extractor.import_prepare_dataset(DATASET_PATH)
        
    # Execute batch processing to extract features
    # feature_extractor.extract_features(dataset, FEATURES_OUTPUT_PATH)

    # Get the filepaths from the images only (without the segmentations)
    dataset = [value['Image'] for value in dataset.values()]

    ######################################################################################################
    ######################################### HISTOGRAM MATCHING #########################################

    # Initialize HistogramMatcher & select histogram matching method
    histogram_matcher = HistogramMatcher(NEW_DATASET_OUTPUT_PATH, 'ExactHistogramMatching')

    # Select Reference Image
    # reference_img = imageio.imread(os.path.join('data', 'test_images', 'Fig2.tif'))
    # target_img = imageio.imread(os.path.join('data', 'test_images', 'Fig1.tif'))
    
    # histogram_matcher.perform_histogram_matching(target_img, reference_img, display=True)

    # Perform histogram matching
    histogram_matcher.perform_batch_histogram_matching(dataset, dataset[0], display=True) # TODO: dataset[0] is temporal, should we automate reference image selection?

    ######################################################################################################
    ########################################### COMPARE RESULTS ##########################################

    

if __name__ == '__main__':

    PARAMETERS_PATH = os.path.join('radiomics_modules', 'Params.yaml')

    DATASET_PATH = os.path.join('data', 'dataset')
    FEATURES_OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')

    NEW_DATASET_OUTPUT_PATH = os.path.join('data', 'new_dataset')
    NEW_FEATURES_OUTPUT_PATH = os.path.join('data', 'new_pyradiomics_extracted_features.csv')

    main()