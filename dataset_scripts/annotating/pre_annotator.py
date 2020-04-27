# will create a csv with a pre annotation for all frames

import argparse
import pickle
import cv2
import os
import numpy as np
from tqdm import tqdm

DATA_PATH = "modelFiles/"

from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *


ap = argparse.ArgumentParser()
ap.add_argument("-fp", "--filesPath", required=True, help="")
ap.add_argument("-rp", "--recordsPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-op", "--outputPath", required=True, help="Caminho para arquivo de saída")
args = vars(ap.parse_args())

faceDetector = CNNOpenCV(args['filesPath'])
roiExtractor = ROI5LandmarksDlib(args['filesPath'], pointsList = ['left', 'right'], scales = [[(1.1, 1)], [(1.1, 1)]], updateRound = 1)
classifier = SigmoidCNNPytorch(args['filesPath'], threshold = 0.1, modelFile = "olho_20x20.pt", fixedSize = (20, 20))


################################# PICKLE/INPUT LOGIC ###################################

pickle_in = open(args["recordsPath"], "rb")
records = pickle.load(pickle_in)
pickle_in.close()

##################################### MAIN LOOP #############################################

with open(args['outputPath'], "w") as f:

    ## HEADER
    f.write("index;frame_start;eyes_status;")
    f.write('\n')

    ## FOR EACH SAVED RECORD/FRAME
    for i, record in tqdm(enumerate(records), "Frames: ", total=len(records)):

        ## RETRIEVING FACE FRAME AND LANDMARKS
        frame = cv2.imread(os.path.join(args['framesDir'], "{}.jpg".format(i)), cv2.IMREAD_COLOR)

        facePoints = faceDetector(frame)
        if facePoints == None:
            print("Cant find face")
            continue
        faceFrame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
        frame = faceFrame.copy()

        eyesPointsFromScales = roiExtractor(frame)
        eyesFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
                        for eyesPoints in eyesPointsFromScales]
        frame = eyesFrames[0].copy()
        output = classifier(eyesFrames[0]) and classifier(eyesFrames[1])

        ## RETRIEVING VALUES TO BE WRITTEN ON CSV FILE
        frameStart = record
        values = [str(i), "%.3f" % frameStart, "closed" if output else "open"]

        cv2.imshow("Face", faceFrame)
        cv2.imshow("ROI", eyesFrames[0])
        cv2.waitKey(1)

        ## WRITING
        f.write(";".join([x for x in values]))
        f.write('\n')
