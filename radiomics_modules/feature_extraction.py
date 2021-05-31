"""
Created on Tue Jul 14 14:11:11 2020

@author: manos
"""

import os
import csv
import collections
from radiomics import featureextractor


def import_dataset():
    cases_dict = {}
    for _, _, files in os.walk(DATASET_PATH):
        for file in files:
            # # Skip case when there is duplicate file in any patient directory
            # if filename in cases_dict.keys() and 'Image' in cases_dict[filename].keys() \
            #         and 'Mask' in cases_dict[filename].keys():
            #     #self.logger.warning('Batch %s: Already exists, skipping this case...', filename)
            #     continue

            file_path = os.path.join(DATASET_PATH, file)

            if "_roi" in file:
                filename = file.rsplit("_")[0]
                if filename in cases_dict.keys():
                    cases_dict[filename].update({'Mask': file_path})
                else:
                    cases_dict[filename] = {'Mask': file_path}
            elif file.endswith(".nii"):
                filename = file.rsplit(".")[0]
                if filename in cases_dict.keys():
                    cases_dict[filename].update({'Image': file_path})
                else:
                    cases_dict[filename] = {'Image': file_path}
    return cases_dict


def extract_features(extractor, cases, output_filepath):
    headers = None
    os.system('rm results.csv')
    for key in cases:
        case = cases[key]
        try:
            image_filepath = case['Image']
            mask_filepath = case['Mask']
        except AttributeError as exception:
            print("Feature extraction error. Missing image or mask.")
            print(exception)
            continue

        feature_vector = collections.OrderedDict(case)
        feature_vector['ID'] = image_filepath.rsplit(".")[0]
        feature_vector['Image'] = os.path.basename(image_filepath)
        feature_vector['Mask'] = os.path.basename(mask_filepath)

        try:
            feature_vector.update(extractor.execute(image_filepath, mask_filepath))
            print("Extracted: ", feature_vector)
            with open(output_filepath, 'a') as outputFile:
                writer = csv.writer(outputFile, lineterminator='\n')
                if headers is None:
                    headers = list(feature_vector.keys())
                    writer.writerow(headers)
                row = []
                for h in headers:
                    row.append(feature_vector.get(h, "N/A"))
                writer.writerow(row)
        except Exception as exception:
            print("Failed to extract features.")
            print(exception)


def main():
    # Get the location of the example settings file
    params = os.path.abspath(PARAMETERS_PATH)
    if not os.path.isfile(params):
        print("Failed to import parameters file.")
        return 
    
    # Import the dataset
    cases = import_dataset()
    if not cases:
        print("Failed to import dataset.")
        return
    
    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    
    # Execute batch processing to extract features
    extract_features(extractor, cases, OUTPUT_PATH)


if __name__ == '__main__':
    
    PARAMETERS_PATH = os.path.join('settings', 'Params.yaml')
    DATASET_PATH = os.path.join('data', 'dataset')
    OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')
    
    main()
