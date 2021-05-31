# Histogram Comparison
# Tsichlaki Styliani mtp209
# Advances in Digital Imaging and Computer Vision
# env Python 3.7

# cmd run instructions:
# (1) cd filepath
# (2) python histogram_comparison.py -d images/

# libs and imports
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

# Construct the argument parser and parse the arguments
# Handle parsing our command line arguments. We only need a single switch,
# --dataset, which is the path to the directory containing our image dataset.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory of images")
args = vars(ap.parse_args())

# Initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves. We initialize two dictionaries.
# The first is index,which stores our color histograms extracted
# from our dataset, with the filename (assumed to be unique)
# as the key, and the histogram as the value.
index = {}
images = {}

# Loop over the image paths
# We utilize glob to grab our image paths and start looping over them
for imagePath in glob.glob(args["dataset"] + "\*.tif"):
    # Extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary.
    filename = imagePath[imagePath.rfind("\\") + 1:]
    image = cv2.imread(imagePath)
    # By default, OpenCV stores images in BGR format rather than RGB.
    # However, we’ll be using matplotlib to display our results, and
    # matplotlib assumes the image is in RGB format. To remedy this,
    # a simple call to cv2.cvtColor is made on Line 27 to convert the
    # image from BGR to RGB.
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update the index.
    # Computing the color histogram.
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # The histogram is normalized.
    hist = cv2.normalize(hist, hist).flatten()
    # It is finally stored in our index dictionary.
    index[filename] = hist

# Initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

# Loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
    # Initialize the results dictionary and the sort direction
    results = {}
    # We start by initializing a reverse variable to False.
    # This variable handles how sorting the results dictionary
    # will be performed. For some similarity functions a LARGER
    # value indicates higher similarity (Correlation and Intersection).
    # And for others, a SMALLER value indicates higher similarity (Chi-Squared and Hellinger).
    reverse = False
    # If we are using the correlation or intersection method, then sort the results in reverse order
    if methodName in ("Correlation", "Intersection"):
        reverse = True
        
    # Loop over the index
    for (k, hist) in index.items():
        # Compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(index["HMres.tif"], hist, method)
        results[k] = d

    # Sort the results
    results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

    # Show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images["HMres.tif"])
    plt.axis("off")

    # Initialize the results figure
    fig = plt.figure("Results: %s" % (methodName))
    fig.suptitle(methodName, fontsize = 20)

    # Loop over the results
    for (i, (v, k)) in enumerate(results):
        # Show the result
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")

# Show the OpenCV methods
plt.show()
