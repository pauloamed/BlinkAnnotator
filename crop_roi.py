import time
import argparse
import pickle
import os
import cv2

from tqdm import tqdm

from blink_utils.models import *


######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--inputPath", required=True, help="")
ap.add_argument("-op", "--outputPath", required=True, help="")
args = ap.parse_args()


faceDetector = OpenCVFaceDetector(args.modelsPath)
extractor = RectangleEyesExtractor()

if not os.path.exists(args.outputPath):
    os.mkdir(args.outputPath)

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

for filePath in tqdm(listdir(args.inputPath)):
    try:
        frame = cv2.imread(os.path.join(args.inputPath, filePath), cv2.IMREAD_COLOR)
    except:
        print("Error while loading " + filePath)
    if frame is None:
        print("Error while loading " + filePath)


    facePoints = faceDetector(frame)
    frame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
    eyesPoints = extractor(frame)
    frame = frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]


    cv2.imwrite(os.path.join(args.outputPath, filePath), frame)
