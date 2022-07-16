# We learn how to utilize a pre-trained Caffee model and use it to classify an 
# image using OpenCV

import numpy as np
import argparse
import time
import cv2

# Construct argument parser
ap = argparse.ArgumentParser()

# Establish four required command line arguments
# The path to the input image
ap.add_argument("-i", "--image", required=True, help="path to input image")
# The path to the Caffe "deploy" prototxt file
ap.add_argument("-p", "prototxt", required=True, help="path to Caffee 'deploy prototxt file")
# The pre-trained Caffee model
ap.add_argument("-m", "--model", required=True, help="path to Caffee pre-trained model")
# The path to ImageNet labels
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")

# Now we parse and store the arguments in a variable for easy access
args = vars(ap.parse_args())

# Load the input image from disk via cv2.imread
image = cv2.imread(args["image"])

# Examples of class label data
# n01440764 tench, Tinca tinca
# n01443537 goldfish, Carassius auratus
# => Unique identifier + Space, some class labels and a new-line
# This makes it easy to parse line by line

# load the class labels from disk into a list
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Result is a list of class labels like this ['tench', 'goldfish']

# Our CNN requires fixed spatial dimensions for our input images
# So we need to ensure it is resized to 224 x 224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input
# After executing this command our "blob" now has the shape: (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1 (224, 224), (104, 117, 123))

# Blob = Binary Large Object, we use cv2.dnn.blobFromImage to perform mean subtraction
# to normalize the input image which results in a known blob shape

