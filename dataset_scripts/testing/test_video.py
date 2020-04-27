from math import floor
import sys
import random
import cv2
from tqdm import tqdm
import os

import argparse

from utils import readCSV, calcMetrics, testVideo
from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *

#######################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
parser.add_argument("-id", "--imagesDir", required = True)
parser.add_argument("-ia", "--imagesAnnotation", required = True)
parser.add_argument("-sf", "--showFrames", required = False)
args = parser.parse_args()

#######################################################################################

showFrames = (args.showFrames == "True")
filesPath = args.filesPath

imagesDir = args.imagesDir
imagesAnnotation = args.imagesAnnotation

faceDetector = UltraLight(args.filesPath, "model.mnn", threshold = 0.8, scale=(1,1))
roiExtractor = ROI5LandmarksDlib(args.filesPath, pointsList = ['left', 'right'], scales = [[(1, 1.1)], [(1, 1.1)]], updateRound = 1)
classifier = HaarOpenCV(args.filesPath)


print(testVideo(faceDetector, roiExtractor, classifier, imagesDir, imagesAnnotation, showFrames))
