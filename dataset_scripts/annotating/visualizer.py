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
ap.add_argument("-s", "--specificFrames", required=False, help="")
args = ap.parse_args()

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

numEntries, originalClasses, frameStarts = readCSV(args.csvPath)

specificFrames = None
if args.specificFrames:
    specificFrames = [int(x) for x in args.specificFrames.split(',')]

for i in tqdm(range(numEntries)):
    frame = cv2.imread(os.path.join(args.framesDir, "{}.jpg".format(i)), cv2.IMREAD_COLOR)

    cv2.putText(frame, originalClasses[i], (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, str(i), (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Cam', frame)
    cv2.waitKey(33)

    if specificFrames and i in specificFrames:
        print("Press ENTER to continue")
        input()
