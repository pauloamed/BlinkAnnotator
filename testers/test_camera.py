from math import floor
import sys
import time
import random
import cv2

import numpy as np

import imutils

import math

import argparse


from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *
from blink_utils.models.face_aligners import *

from blink_utils.models import PreRotator, WindowFilter

from blink_utils import BlinkLengthEstimator



parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
args = parser.parse_args()

filesPath = args.filesPath

faceDetector = UltraLightONNX(args.filesPath, "version-slim-320_320_simplified.onnx", threshold = 0.8, inputSize=320)

aligner = Aligner5LandmarksDlib(args.filesPath)

roiExtractor = ROI5LandmarksDlib(args.filesPath, pointsList = ['left', 'right'], scales = [[(1.1, 1.1)], [(1.1, 1.1)]], updateRound = 1)
classifier = SigmoidCNNPytorch(args.filesPath, threshold = 0.8, modelFile = "midelo_30x30.pt", fixedSize = (30, 30), normalize=True, closedId=0)

lengthEstimator = BlinkLengthEstimator()
prerot = PreRotator()
filt = WindowFilter('hamming', 5, threshold=0.5)


# videoCapture = None
# try:
#     videoCapture = cv2.VideoCapture(0)
# except:
#     print('Não foi possível carregar o dispositivo de captura de vídeo')
#     exit()


showTime = False

frameCount = 0
lastSec = math.floor(time.time())

total = 0
timeFace = 0
timeRoi = 0
timeClass = 0

while True:
    total += 1
    _, frame = videoCapture.read()
    frame = prerot(frame)
    cv2.imshow("a", frame)

    # startFaceDetector = time.time()
    # facePoints = faceDetector(frame)
    # timeFace += time.time() - startFaceDetector
    # if showTime: print("Face Detector: {:.4f}".format(time.time() - startFaceDetector))
    # if facePoints == None:
    #     print("Cant find face")
    #     continue
    # faceFrame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
    # frame = faceFrame
    #
    # angle = aligner(frame)
    # frame = aligner.rotateFrame(frame)
    # prerot.update(angle, facePoints)
    #
    # cv2.imshow("b", frame)

    eyesFrames = None
    if roiExtractor:
        startRoiExtractor = time.time()
        eyesPointsFromScales = roiExtractor(frame)
        timeRoi += time.time() - startRoiExtractor
        if showTime: print("Roi Extractor: {:.4f}".format(time.time() - startRoiExtractor))
        eyesFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
                        for eyesPoints in eyesPointsFromScales]
        frame = eyesFrames[0]

    cv2.imshow("c", frame)

    startClassifier = time.time()
    output = classifier(eyesFrames[0]) * classifier(eyesFrames[1])
    output = filt(output)

    lengthEstimator.updateStatus(output)

    print(lengthEstimator.getLastBlink())

    timeClass += time.time() - startClassifier
    if showTime: print("Classifer: {:.4f}".format(time.time() - startClassifier))



    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" if output else "")

    frameCount += 1
    if math.floor(time.time()) > lastSec:
        print("FPS: {}".format(frameCount))
        lastSec = math.floor(time.time())
        frameCount = 0

    # logica do opencv, aperta q pra fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(timeFace / total, timeRoi / total, timeClass / total)
