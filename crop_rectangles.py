import time
import argparse
import pickle
import os
import cv2

from eyes_extractor.RectangleEyesExtractor import RectangleEyesExtractor
from eyes_extractor.OpenCVFaceDetector import OpenCVFaceDetector

from tqdm import tqdm

#######################################################################################

def readCSV(inputPath):
    numEntries, originalClasses, frameStarts = None, [], []
    with open(inputPath) as file:
        lines = list(file.readlines())
        header = lines[0].split(';')

        frameStartIndex = header.index('frame_start')
        isBlinkingIndex = header.index('eyes_status')

        lines = lines[1:]
        numEntries = len(lines)

        for line in lines:
            line = line.split(';')
            frameStarts.append(line[frameStartIndex].strip())
            originalClasses.append(line[isBlinkingIndex].strip())

    return numEntries, originalClasses, frameStarts

######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-cp", "--csvPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-mp", "--modelsPath", required=True, help="")
ap.add_argument("-op", "--outputPath", required=True, help="")
args = ap.parse_args()

faceDetector = OpenCVFaceDetector(args.modelsPath)
extractor = RectangleEyesExtractor()

if not os.path.exists(args.outputPath):
    os.mkdir(args.outputPath)

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

numEntries, originalClasses, frameStarts = readCSV(args.csvPath)


fileSpecific = args.framesDir.split('/')[-1]


closedCount = openCount = 0
for i in tqdm(range(numEntries)):
    frame = cv2.imread(os.path.join(args.framesDir, "{}.jpg".format(i)), cv2.IMREAD_COLOR)
    facePoints = faceDetector.detect(frame)
    frame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
    eyesPoints = extractor.extract(frame)
    frame = frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
    text = originalClasses[i]

    fileName = fileSpecific + "_"
    if originalClasses[i] == 'closed':
        fileName +=  text + "_{}.jpg".format(closedCount)
        closedCount += 1
    else:
        fileName += text + "_{}.jpg".format(openCount)
        openCount += 1

    cv2.imshow("a", frame)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(args.outputPath, fileName), frame)
