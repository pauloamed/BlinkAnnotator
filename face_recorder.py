import time
import argparse
import pickle
import os
from pynput import keyboard
import cv2

eyesClosed = False
success = True

def onPress(key):
    global eyesClosed
    try:
        if key.char == "n":
            eyesClosed = True
    except:
        pass

def onRelease(key):
    global eyesClosed, success
    try:
        if key.char == "q":
            success = False
            return False
        if key.char == "n":
            eyesClosed = False
    except:
        pass


listener = keyboard.Listener(
    on_press=onPress,
    on_release=onRelease)

listener.start()


######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-rp", "--recordsPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
args = vars(ap.parse_args())

######################### VIDEO CAPTURE + FACE DETECT PREP #############################

videoCapture = None
try:
    videoCapture = cv2.VideoCapture(0)
except:
    print('Não foi possível carregar o dispositivo de captura de vídeo')
    exit()

cnnThreshold = 0.9
dataPath = "modelFiles/"
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
cnnNet = cv2.dnn.readNetFromTensorflow(os.path.join(dataPath, modelFile), os.path.join(dataPath, configFile))

_, frame = videoCapture.read()
facePoints = (0, frame.shape[0], 0, frame.shape[1])

####################################### OUTPUT PREP ###################################

if not os.path.exists(args['framesDir']):
    os.mkdir(args['framesDir'])

##################################### AUX VARS INIT ##################################

loopStart = time.time()
records = []
frameRate = -1

##################################### MAIN LOOP ######################################

while (time.time() - loopStart) <= 60 and success: ## EXECUTE LOOP FOR 60 SECS
    frameStart = time.time()

    ## FRAME CAPTURE AND FACE FRAME EXTRACTION


    _, frame = videoCapture.read()
    frame = cv2.resize(frame, (320, 240))

    newFacePoints = None
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    cnnNet.setInput(blob) # blob sera a entrada da rede
    detections = cnnNet.forward() # deteccoes feitas na rede
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > cnnThreshold:
            x1 = max(int(detections[0, 0, i, 3] * frame.shape[1]), 0)
            y1 = max(int(detections[0, 0, i, 4] * frame.shape[0]), 0)
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])
            newFacePoints = (y1, y2, x1, x2)
            break

    if newFacePoints:
        facePoints = newFacePoints
    frame = frame[facePoints[0]:facePoints[1],facePoints[2]:facePoints[3]]

    ## READING IF KEY WAS PRESSED

    cv2.waitKey(1)
    ## SAVING OUTPUT

    cv2.imwrite(os.path.join(args['framesDir'], "{}.jpg".format(len(records))), frame)
    records.append((eyesClosed, frameStart - loopStart))

    ## FEEDBACK LOGIC


    cv2.putText(frame, "FPS: {}".format(frameRate), (15, int(frame.shape[0] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(frame, "closed" if eyesClosed else "open", (15, int(frame.shape[0] * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.putText(frame, "%.2f" % (time.time() - loopStart), (15, int(frame.shape[0] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('Cam', frame)

    ## FRAME RATE LOGIC

    frameEnd = time.time()
    frameDuration = frameEnd - frameStart
    frameRate = int(1 / frameDuration)


############################ PICKLE/OUTPUT LOGIC #############################

if success:
    pickle_out = open(args['recordsPath'], "wb")
    pickle.dump(records, pickle_out)
    pickle_out.close()
else:
    import shutil
    shutil.rmtree(args['framesDir'], ignore_errors=True)
