
import time
import argparse
import pickle
import os
import cv2

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

args = ap.parse_args()

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

numEntries, originalClasses, frameStarts = readCSV(args.csvPath)


fileSpecific = args.framesDir.split('/')[-1]


closedCount = openCount = 0
for i in tqdm(range(numEntries)):
    fileName = os.path.join(args.framesDir, "{}.jpg".format(i))

    newFileName = None
    if originalClasses[i] == 'closed':
        newFileName = "{}_closed_{}.jpg".format(i, closedCount)
        closedCount += 1
    else:
        newFileName = "{}_open_{}.jpg".format(i, openCount)
        openCount += 1

    newFileName = os.path.join(args.framesDir, newFileName)

    os.rename(fileName, newFileName)
