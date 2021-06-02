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
    feature_extractor.extract_features(dataset, FEATURES_OUTPUT_PATH)

    # Get the filepaths from the images only (without the segmentations)
    dataset_images = [value['Image'] for value in dataset.values()]


    ######################################################################################################
    ######################################### HISTOGRAM MATCHING #########################################

    # Initialize HistogramMatcher & select histogram matching method
    histogram_matcher = HistogramMatcher(NEW_DATASET_OUTPUT_PATH, 'ExactHistogramMatching')

    # Select Reference Image
    # reference_img = imageio.imread(os.path.join('data', 'test_images', 'Fig2.tif'))
    # target_img = imageio.imread(os.path.join('data', 'test_images', 'Fig1.tif'))
    
    # histogram_matcher.perform_histogram_matching(target_img, reference_img, display=True)

    # Perform histogram matching
    # histogram_matcher.perform_batch_histogram_matching(dataset_images, dataset_images[0], display=True) # TODO: dataset[0] is temporal, should we automate reference image selection?


    ######################################################################################################
    ##################################### EXTRACT RADIOMICS FEATURES #####################################
    #####################################    FROM THE NEW DATASET    #####################################

    # Import the dataset
    new_dataset = feature_extractor.import_prepare_dataset(NEW_DATASET_OUTPUT_PATH)

    # Copy the segmentations from the old dataset to the new one
    for key, value in new_dataset.items():
        # Get the image path, replace it with the image path from the old dataset
        # and add _roi in order to create the mask path
        path = value['Image'].split('.')                                 # split the path into a list
        path[0] = path[0].replace(NEW_DATASET_OUTPUT_PATH, DATASET_PATH) # replace the new path with the old one
        path.insert(1, '_roi.')                                          # append _roi
        path = ''.join(path)                                             # join the list elements into a string

        # Add mask path from the old dataset to new dataset dictionary
        new_dataset[key]['Mask'] = path

    # Execute batch processing to extract features
    feature_extractor.extract_features(new_dataset, NEW_FEATURES_OUTPUT_PATH)

    # Get the filepaths from the images only (without the segmentations)
    new_dataset = [value['Image'] for value in new_dataset.values()]

    
    ######################################################################################################
    ########################################### COMPARE RESULTS ##########################################

    

if __name__ == '__main__':

    PARAMETERS_PATH = os.path.join('radiomics_modules', 'Params.yaml')

    DATASET_PATH = os.path.join('data', 'dataset')
    FEATURES_OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')

    NEW_DATASET_OUTPUT_PATH = os.path.join('data', 'new_dataset')
    NEW_FEATURES_OUTPUT_PATH = os.path.join('data', 'new_pyradiomics_extracted_features.csv')

    main()