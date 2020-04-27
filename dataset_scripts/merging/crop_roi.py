import time
import argparse
import pickle
import os
import cv2

from tqdm import tqdm

from blink_utils.models import *


######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-mp", "--modelsPath", required=True)
ap.add_argument("-ip", "--inputPath", required=True, help="")
ap.add_argument("-op", "--outputPath", required=True, help="")
args = ap.parse_args()


faceDetector = UltraLightONNX(args.modelsPath, "version-RFB-320_320_simplified.onnx", threshold = 0.8, inputSize=320)
extractor = ROI5LandmarksDlib(args.modelsPath, pointsList = ['left', 'right'], scales = [[(1.1, 1)], [(1.1, 1)]], updateRound = 1)

if not os.path.exists(args.outputPath):
    os.mkdir(args.outputPath)

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

for filePath in tqdm(os.listdir(args.inputPath)):
    try:
        frame = cv2.imread(os.path.join(args.inputPath, filePath), cv2.IMREAD_COLOR)
    except:
        print("Error while loading " + filePath)
    if frame is None:
        print("Error while loading " + filePath)


    facePoints = faceDetector(frame)
    if facePoints is None:
        continue

    frame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]


    eyesPoints = extractor(frame)
    if eyesPoints is None:
        continue
    for i, eyePoints in enumerate(eyesPoints):
        eyeFrame = frame[eyePoints['y1']:eyePoints['y2'], eyePoints['x1']:eyePoints['x2']]

        # cv2.imshow("b", eyeFrame)
        # cv2.waitKey(1000)

        cv2.imwrite(os.path.join(args.outputPath, filePath + "_{}.jpg".format(i)), eyeFrame)
