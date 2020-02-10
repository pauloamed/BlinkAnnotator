import time
import argparse
import pickle
import os

import cv2
from OpenCVFaceDetector import OpenCVFaceDetector

##################################### AUX FUNCS ########################################

def getFaceFrame(frame, facePoints):
    faceFrame = frame[facePoints['y1']:facePoints['y2'],
        facePoints['x1']:facePoints['x2']]

    return faceFrame

######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataPath", required=True, help="Caminho para arquivos do detector de face")
ap.add_argument("-rp", "--recordsPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-od", "--outDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
args = vars(ap.parse_args())

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

videoCapture = None
try:
    videoCapture = cv2.VideoCapture(0)
except:
    print('Não foi possível carregar o dispositivo de captura de vídeo')
    exit()

faceDetector = OpenCVFaceDetector(args["dataPath"])

_, frame = videoCapture.read()
facePoints = faceDetector.detect(frame)

####################################### OUTPUT PREP ###################################

if not os.path.exists(args['outDir']):
    os.mkdir(args['outDir'])

##################################### AUX VARS INIT ##################################

loopStart = time.time()
isBlinking = False
records = []


##################################### MAIN LOOP ######################################

while (time.time() - loopStart) <= 60: ## EXECUTE LOOP FOR 60 SECS

    frameStart = time.time()
    frameRate = -1

    ## FRAME CAPTURE AND FACE FRAME EXTRACTION

    _, frame = videoCapture.read()
    newFacePoints = faceDetector.detect(frame)
    if newFacePoints:
        facePoints = newFacePoints
    frame = getFaceFrame(frame, facePoints)

    ## READING IF KEY WAS PRESSED

    if cv2.waitKey(1) == ord('n'):
        isBlinking = not isBlinking

    ## SAVING OUTPUT

    cv2.imwrite(args['outDir'] + "/{}.jpg".format(len(records)), frame)
    records.append((isBlinking, frameStart - loopStart))

    ## FEEDBACK LOGIC

    cv2.putText(frame, "FPS: {}".format(frameRate), (15, int(frame.shape[0] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, "open" if isBlinking else "closed", (15, int(frame.shape[0] * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, "%.2f" % (time.time() - loopStart), (15, int(frame.shape[0] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('Cam', frame)

    ## FRAME RATE LOGIC

    frameEnd = time.time()
    frameDuration = frameEnd - frameStart
    frameRate = int(1 / frameDuration)


############################ PICKLE/OUTPUT LOGIC #############################

pickle_out = open(args['recordsPath'], "wb")
pickle.dump(records, pickle_out)
pickle_out.close()
