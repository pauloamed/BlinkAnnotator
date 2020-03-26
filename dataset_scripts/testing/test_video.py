from math import floor
import sys
import random
import cv2
from tqdm import tqdm
import os

import argparse

from utils import loadModels, readCSV, calcMetrics, testVideo

#######################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
parser.add_argument("-fd", "--face_detector", required = True)
parser.add_argument("-re", "--roi_extractor", required = False)
parser.add_argument("-c", "--classifier", required = True)
parser.add_argument("-id", "--imagesDir", required = True)
parser.add_argument("-ia", "--imagesAnnotation", required = True)
parser.add_argument("-sf", "--showFrames", required = False)
args = parser.parse_args()

#######################################################################################

showFrames = (args.showFrames == "True")
filesPath = args.filesPath

imagesDir = args.imagesDir
imagesAnnotation = args.imagesAnnotation

faceDetector, roiExtractor, classifier = loadModels(args, filesPath)


print(testVideo(faceDetector, roiExtractor, classifier, imagesDir, imagesAnnotation, showFrames))
