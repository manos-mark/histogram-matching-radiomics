"""
Created on Tue Jul 14 14:11:11 2020

@author: manos
"""

import os
import csv
import collections
# from radiomics import featureextractor
import skimage.io
from skimage.feature import greycomatrix, greycoprops
import numpy as np


class FeatureExtractor:

    def __init__(self, parameters_path):
        # Get the location of the example settings file
        self.parameters_path = os.path.abspath(parameters_path)

        if not os.path.isfile(self.parameters_path):
            raise FileNotFoundError("Failed to import parameters file.")
        
        # Initialize feature extractor using the settings file
        # self.extractor = featureextractor.RadiomicsFeatureExtractor(self.parameters_path)

    def extract_features(self, cases, output_filepath):
        headers = None

        if os.path.isfile(output_filepath):
            os.system('rm ' + output_filepath)

        for key in cases:
            case = cases[key]
            try:
                image_filepath = case['Image']
                mask_filepath = None#case['Mask']
            except AttributeError as exception:
                print("Feature extraction error. Missing image or mask.")
                print(exception)
                continue

            feature_vector = collections.OrderedDict(case)
            feature_vector['ID'] = image_filepath.rsplit(".")[0].rsplit("/")[-1]
            # feature_vector['ID'] = image_filepath
            feature_vector['Image'] = os.path.basename(image_filepath)
            # feature_vector['Mask'] = os.path.basename(mask_filepath)

            try:
                # feature_vector.update(self.extractor.execute(image_filepath, mask_filepath))
                image = skimage.io.imread(image_filepath)
                image = image[:, :, 0]
                assert type(image) == np.ndarray, 'Input must be NumPy Array'

                features = {'contrast', 'homogeneity', 'energy'}
                feature_list = dict()

                for feature in features:
                    g = greycoprops(greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]), feature)
                    feature_list[feature] = g

                feature_vector.update(feature_list)

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
