import os
import argparse
import cv2
import time

from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *

def maybeCreateDir(path):
    if not os.path.exists(path):
        os.mkdir(path)




parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
parser.add_argument("-op", "--outputPath", required = True)
args = parser.parse_args()

outputPath = args.outputPath
videosPath = os.path.join(outputPath, "videos")
facesPath = os.path.join(outputPath, "faces")

maybeCreateDir(outputPath)
maybeCreateDir(facesPath)
maybeCreateDir(videosPath)

faceDetector = UltraLightONNX(args.filesPath, "version-slim-320_320_simplified.onnx", threshold = 0.97, inputSize=320)
roiExtractor = ROI5LandmarksDlib(args.filesPath, pointsList = ['left', 'right'], scales = [[(1.1, 1)], [(1.1, 1)]], updateRound = 1)
classifier = SigmoidCNNPytorch(args.filesPath, threshold = 0.1, modelFile = "modela.pt", fixedSize = (30, 30), normalize=True)

j = 0
while True:
    if len(os.listdir(videosPath)) == 0:
        print("Waiting due to no files")
        time.sleep(10)
        continue

    fileName, fileExtension = None, None
    for file in os.listdir(videosPath):
        fileName, fileExtension = os.path.splitext(file)
        if fileExtension == '.webm': break

    if fileExtension != '.webm':
        print("Waiting due to no webm files")
        time.sleep(10)
        continue

    filePath = os.path.join(videosPath, file)
    print("Extracting " + filePath)

    vidcap = cv2.VideoCapture(filePath)
    i = 0
    contClosed = 0
    contOpen = 0
    while True:
        success, image = vidcap.read()
        if success:
            facePoints = faceDetector(image)
            if facePoints:
                image = image[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]

                if image is None or len(image) == 0:
                    continue
                if image.shape[0] < 150 or image.shape[1] < 150:
                    continue

                frame = image.copy()
                eyesPointsFromScales = roiExtractor(frame)
                if eyesPointsFromScales is None:
                    continue
                eyesFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
                                for eyesPoints in eyesPointsFromScales]
                output = classifier(eyesFrames[0])

                if output:
                    contClosed += 1
                else:
                    if contOpen == 50:
                        continue
                    contOpen += 1

                faceFilePath = fileName + "_" + str(i)
                cv2.imwrite(os.path.join(facesPath, "{}.jpg".format(faceFilePath)), image)
                print("{}: ({}/{})".format(j, i + 1, 100))
                i += 1
                if i > 100:
                    break
        else:
            break

    vidcap.release()
    os.remove(filePath)
    j += 1
