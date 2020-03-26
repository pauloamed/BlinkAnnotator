from math import floor
import sys
import time
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
parser.add_argument("-dp", "--datasetPath", required = True)
parser.add_argument("-sf", "--showFrames", required = False)
args = parser.parse_args()

#######################################################################################

showFrames = True
if args.showFrames:
    showFrames = True if args.showFrames == "true" else False

modelsPath = args.modelsPath
datasetPath = args.datasetPath


videosToTest = [filePath for filePath in os.listdir(datasetPath) if '.' not in filePath]

print(len(videosToTest), videosToTest)
mask = '11111111111111111'

faceDetector, openEyesClassifer = loadModels(args, modelsPath)
detector = BlinkClassifer(faceDetector, openEyesClassifer)

output = ""
for i in range(len(mask)):
    if mask[i] == '1':
        imagesDir = os.path.join(datasetPath, videosToTest[i])
        imagesAnnotation = os.path.join(datasetPath, videosToTest[i] + ".csv")
        output += testVideo(detector, imagesDir, imagesAnnotation, showFrames)
