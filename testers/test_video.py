from math import floor
import sys
import random
import cv2
from tqdm import tqdm
import os

from blinkClassifier import BlinkClassifer

import argparse

from utils import loadModels, readCSV, calcMetrics, testVideo

#######################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-mp", "--modelsPath", required = True)
parser.add_argument("-fd", "--faceDetector", required = True)
parser.add_argument("-ee", "--eyesExtractor", required = False)
parser.add_argument("-ef", "--eyesFrameClassifier", required = False)
parser.add_argument("-oc", "--openEyesClassifer", required = False)
parser.add_argument("-id", "--imagesDir", required = True)
parser.add_argument("-ia", "--imagesAnnotation", required = True)
parser.add_argument("-sf", "--showFrames", required = False)
args = parser.parse_args()

#######################################################################################

showFrames = (args.showFrames == "True")
modelsPath = args.modelsPath

imagesDir = args.imagesDir
imagesAnnotation = args.imagesAnnotation

faceDetector, openEyesClassifer = loadModels(args, modelsPath)
detector = BlinkClassifer(faceDetector, openEyesClassifer)


print(testVideo(detector, imagesDir, imagesAnnotation, showFrames))
