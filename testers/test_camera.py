from math import floor
import sys
import time
import random
import cv2

import argparse

from utils import loadModels

from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
args = parser.parse_args()

filesPath = args.filesPath

faceDetector = CNNOpenCV(args.filesPath)
roiExtractor = ROI68LandmarksDlib(args.filesPath, pointsList = ['sboth'], scales = [[(3, 3)]], updateRound = 1)
classifier = HaarOpenCV(args.filesPath)

aligner = Aligner5LandmarksDlib(args.filesPath)


videoCapture = None
try:
    videoCapture = cv2.VideoCapture(0)
except:
    print('Não foi possível carregar o dispositivo de captura de vídeo')
    exit()


showTime = False

while True:
    loopStart = time.time()
    _, frame = videoCapture.read()

    startFaceDetector = time.time()
    facePoints = faceDetector(frame)
    if showTime: print("Face Detector: {:.4f}".format(time.time() - startFaceDetector))
    if facePoints == None:
        print("Cant find face")
        continue
    faceFrame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
    frame = faceFrame

    if roiExtractor:
        startRoiExtractor = time.time()
        eyesPointsFromScales = roiExtractor(frame)
        if showTime: print("Roi Extractor: {:.4f}".format(time.time() - startRoiExtractor))
        eyesFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
                        for eyesPoints in eyesPointsFromScales]
        frame = eyesFrames[0]

    cv2.imshow("a", frame)

    startClassifier = time.time()
    output = classifier(frame)
    if showTime: print("Classifer: {:.4f}".format(time.time() - startClassifier))


    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" if output else "")
    if showTime:
        fps = 1 / (time.time() - loopStart)
        print("FPS: {}".format(fps), time.time() - loopStart)

    # logica do opencv, aperta q pra fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        continue
