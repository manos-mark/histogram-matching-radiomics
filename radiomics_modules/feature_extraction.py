"""
Created on Tue Jul 14 14:11:11 2020

@author: manos
"""

import os
import csv
import collections
from radiomics import featureextractor


class FeatureExtractor:

    def __init__(self, parameters_path):
        # Get the location of the example settings file
        self.parameters_path = os.path.abspath(parameters_path)

        if not os.path.isfile(self.parameters_path):
            raise FileNotFoundError("Failed to import parameters file.")
        
        # Initialize feature extractor using the settings file
        self.extractor = featureextractor.RadiomicsFeatureExtractor(self.parameters_path)


    def import_prepare_dataset(self, dataset_path):
        cases_dict = {}
        import glob
        dirnames = glob.glob(os.path.join(dataset_path, "*", ""))
        
        for dir in dirnames:
            filenames = glob.glob(os.path.join(dir, "*.tif"))

            for file in filenames:

                if "_mask" in file:
                    filename = file.rsplit("_")[:-1]
                    filename = '_'.join(filename)
                    if filename in cases_dict.keys():
                        cases_dict[filename].update({'Mask': file})
                    else:
                        cases_dict[filename] = {'Mask': file}

                elif file.endswith(".tif"):
                    filename = file.rsplit(".")[:-1]
                    # filename = file.rsplit("_")[:-1]
                    filename = ''.join(filename)
                    print(filename)
                    if filename in cases_dict.keys():
                        cases_dict[filename].update({'Image': file})
                    else:
                        cases_dict[filename] = {'Image': file}

            if not cases_dict:
                raise FileNotFoundError("Failed to import dataset.")
                
        return cases_dict


    def extract_features(self, cases, output_filepath):
        headers = None

        if os.path.isfile(output_filepath):
            os.system('rm ' + output_filepath)

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
            # feature_vector['ID'] = image_filepath.rsplit(".")[0]
            feature_vector['ID'] = image_filepath.rsplit("_")[-1]
            feature_vector['Image'] = os.path.basename(image_filepath)
            feature_vector['Mask'] = os.path.basename(mask_filepath)

            try:
                feature_vector.update(self.extractor.execute(image_filepath, mask_filepath))
                # print("Extracted: ", feature_vector)
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
