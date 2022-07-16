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

