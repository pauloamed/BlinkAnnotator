'''
the faces will be aligned
for each eye, K differentt crops will be taken
K is an arg input

'''


from math import floor
import sys
import time
import random
import os
import cv2

import numpy as np

import imutils

import math

import argparse


from blink_utils.models.roi_extractors import *
from blink_utils.models.face_aligners import *


def getRandom5Scales(k):
    scales = []
    for i in range(k):
        scaleVer = random.randint(100, 200) / 100
        scaleHor = random.randint(100, 200) / 100
        scales.append((scaleHor, scaleVer))
    return scales

def getRandom68Scales(k):
    scales = []
    for i in range(k):
        scaleVer = random.randint(100, 200) / 100
        scaleHor = random.randint(100, 150) / 100
        scales.append((scaleHor, scaleVer))
    return scales


datasetDir = 'ClosedFace'
datasetDir = 'OpenFace'


parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
args = parser.parse_args()

filesPath = args.filesPath

aligner = Aligner5LandmarksDlib(args.filesPath)

scales = [(1, 1)]
roi68Extractor = ROI68LandmarksDlib(args.filesPath, pointsList = ['sleft', 'sright'], scales = [scales, scales], updateRound = 1)
roi5Extractor = ROI5LandmarksDlib(args.filesPath, pointsList = ['left', 'right'], scales = [scales, scales], updateRound = 1)

for filePath in os.listdir(datasetDir):
    filePath = os.path.join(datasetDir, filePath)

    frame = cv2.imread(filePath, cv2.IMREAD_COLOR)

    cv2.imshow("Frame", frame)

    angle = aligner(frame)

    alignedFrame = imutils.rotate_bound(frame, angle * 180 / np.pi)

    # eyesAlignedPoints = roi68Extractor(alignedFrame)
    # eyesAlignedPoints += roi5Extractor(alignedFrame)

    for i in range(len(roi5Extractor.scales)):
        roi5Extractor.scales[i] = getRandom5Scales(2)
    for i in range(len(roi5Extractor.scales)):
        roi68Extractor.scales[i] = getRandom68Scales(2)

    print(roi5Extractor.scales)
    print(roi68Extractor.scales)

    eyesUnalignedPoints = roi68Extractor(frame)
    eyesUnalignedPoints += roi5Extractor(frame)



    # eyesAlignedFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
    #         for eyesPoints in eyesAlignedPoints]
    eyesUnlignedFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
            for eyesPoints in eyesUnalignedPoints]

    for i, eyesFrame in enumerate(eyesUnlignedFrames):
        cv2.imshow(str(i), eyesFrame)

    cv2.waitKey(1000)
