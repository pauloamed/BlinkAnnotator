# will create a csv with a pre annotation for all frames

import argparse
import pickle
import dlib
import cv2
import os
import numpy as np
from tqdm import tqdm

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

DATA_PATH = "modelFiles/"
LEFT_EYE_INDEXES = range(36, 42)
RIGHT_EYE_INDEXES = range(42, 48)

##########################################################################################

import cv2
from math import hypot


modelFile = os.path.join(DATA_PATH, "opencv_face_detector_uint8.pb")
configFile = os.path.join(DATA_PATH, "opencv_face_detector.pbtxt")

cnnNet = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
cnnThreshold = 0.9

def facePunct(face, frameCenter):
    area = abs(face['x2'] - face['x1']) * abs(face['y2'] - face['y1'])
    faceCenter = ((face['x1'] + face['x2']) // 2, (face['y1'] + face['y2']) // 2)
    distCenters = hypot(frameCenter[0] - faceCenter[0], frameCenter[1] - faceCenter[1])
    confidence = face['conf']
    return area * 1/distCenters * confidence


def getFace(cnnNet, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    cnnNet.setInput(blob) # blob sera a entrada da rede
    detections = cnnNet.forward() # deteccoes feitas na rede

    aboveThresholdFaces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > cnnThreshold:
            face = {
                'conf' : confidence,
                'x1' : max(int(detections[0, 0, i, 3] * frame.shape[1]), 0),
                'y1' : max(int(detections[0, 0, i, 4] * frame.shape[0]), 0),
                'x2' : int(detections[0, 0, i, 5] * frame.shape[1]),
                'y2' : int(detections[0, 0, i, 6] * frame.shape[0]),
            }
            aboveThresholdFaces.append(face)

    frameCenter = (frame.shape[0]//2, frame.shape[1]//2)

    bestFace = max(aboveThresholdFaces, key=lambda x : facePunct(x, frameCenter))

    return bestFace

######################################## AUX #############################################

def getEyeImage(gray, eyeIndexes, landmarks):
    eyePoints = [(landmarks.part(i).x, landmarks.part(i).y) for i in eyeIndexes]

    (x, y, w, h) = cv2.boundingRect(np.array([eyePoints]))
    return gray[y:y + h, x:x + w]

##################################### CNN MODEL ##########################################
num_cores = 4

num_CPU = 1
num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU})

session = tf.Session(config=config)
K.set_session(session)

model = load_model(os.path.join(DATA_PATH, "weights.149-0.01.hdf5"))

def predict_eye_state(model, image):
    image = cv2.resize(image, (20, 10))
    image = image.astype(dtype=np.float32)

    image_batch = np.reshape(image, (1, 10, 20, 1))
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)

    return np.argmax( model.predict(image_batch)[0] )


def checkIfBlinking(model, leftEye, rightEye):
    leftEyePredict = predict_eye_state(model, leftEye)
    rightEyePredict = predict_eye_state(model, rightEye)

    return (leftEyePredict == 0) or (rightEyePredict == 0)

######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-rp", "--recordsPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-op", "--outputPath", required=True, help="Caminho para arquivo de saída")
args = vars(ap.parse_args())

############################## LANDMARKS PREDICTOR PREP ################################

predictor = dlib.shape_predictor(os.path.join(DATA_PATH, "shape_predictor_68_face_landmarks.dat"))

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
        if frame is None:
            raise Exception("Não foi possível abrir frame {}".format(i))

        facePuncts = getFace(cnnNet, frame)
        faceFrame = frame[facePuncts['y1']:facePuncts['y2'], facePuncts['x1']:facePuncts['x2']]
        gray = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2GRAY)

        landmarks = predictor(faceFrame, dlib.rectangle(0, 0, faceFrame.shape[1], faceFrame.shape[0]))

        leftEye = getEyeImage(gray, LEFT_EYE_INDEXES, landmarks)
        rightEye = getEyeImage(gray, RIGHT_EYE_INDEXES, landmarks)

        isBlinking = checkIfBlinking(model, leftEye, rightEye)

        ## RETRIEVING VALUES TO BE WRITTEN ON CSV FILE
        frameStart = record
        values = [str(i), "%.3f" % frameStart, "closed" if isBlinking else "open"]

        cv2.imshow("Face", gray)
        cv2.waitKey(1)

        ## WRITING
        f.write(";".join([x for x in values]))
        f.write('\n')
