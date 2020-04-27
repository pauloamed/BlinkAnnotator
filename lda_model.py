from math import floor
import sys
import time
import cv2
import dlib
import imutils
from imutils import face_utils


import joblib

import numpy as np

import math

import argparse

import matplotlib.pyplot as plt


from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *

from blink_utils.models.face_aligners import *

from blink_utils import BlinkLengthEstimator


parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--filesPath", required = True)
parser.add_argument("-emp", "--EyeModelPath", default='./modelFiles/lda_model.pkl')
parser.add_argument("-sp", "--ShapePath", default='./modelFiles/shape_predictor_68_face_landmarks.dat')
args = parser.parse_args()

filesPath = args.filesPath

faceDetector = UltraLightONNX(args.filesPath, "version-slim-320_128_simplified.onnx", threshold = 0.8, inputSize=128)

lda_model = joblib.load(args.EyeModelPath)

aligner = Aligner5LandmarksDlib(args.filesPath)

shape_predictor = args.ShapePath
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# print(face_utils.FACIAL_LANDMARKS_IDXS["left_eye"])

def eye_point_extract(Lefteye,Righteye):
  eye_points = []
  for point in Lefteye:
    x = point[0]
    y = point[1]
    eye_points.append(x)
    eye_points.append(y)

  for point in Righteye:
    x = point[0]
    y = point[1]
    eye_points.append(x)
    eye_points.append(y)
  return eye_points


def eye_colect_points(image):
    print(image.shape)
    points = None
    # image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    reacts = detector(gray,0)

    width, height = image.shape[1::-1]
    shape=predictor(image,dlib.rectangle(0, 0, width, height))
    shape = face_utils.shape_to_np(shape)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]


    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

    points = eye_point_extract(leftEye,rightEye)

    return points, image


videoCapture = None
try:
    videoCapture = cv2.VideoCapture(0)
except:
    print('Não foi possível carregar o dispositivo de captura de vídeo')
    exit()

showTime = False
LDAblinks=[]
frameCount = 0
lastSec = math.floor(time.time())

total = 0
timeFace = 0
timeRoi = 0
timeClass = 0

while True:
    total += 1
    _, frame = videoCapture.read()
    cv2.imshow("a", frame)

    startFaceDetector = time.time()
    facePoints = faceDetector(frame)
    timeFace += time.time() - startFaceDetector
    if facePoints == None:
        print("Cant find face")
        continue
    faceFrame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
    frame = faceFrame
    angle = aligner(frame)

    frame = imutils.rotate_bound(frame, angle * 180 / np.pi)

    eye_points, frame = eye_colect_points(frame)
    print(eye_points)

    if eye_points is not None:
      eye_points = np.asarray(eye_points)
      eye_points = eye_points.reshape(1,-1)

      eye_calssifier = lda_model.predict(eye_points)
      LDAblinks.append(eye_calssifier)
      print( eye_calssifier )
    else: print('cant find eyes points')

    cv2.imshow("b", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

LDAblinks = np.asarray(LDAblinks).flatten()

def segmentation(blinks):
  new_blinks = blinks.copy()
  count = 0
  long_blink_tresh = 3
  long_blink_value = 2
  for idx,image in enumerate(blinks):
      if image == 1:
          count += 1
      else:
          if count > long_blink_tresh:
              for blink in range(idx-count,idx+1):
                new_blinks[blink] = long_blink_value
          count = 0
  return new_blinks



fig = plt.figure(figsize=(10,5))
plt.title('Long and Short Blinks')
plt.xlabel('Index of Frames')
plt.yticks([1,2],['Short', 'Long'])
plt.plot(segmentation(LDAblinks),color='red')
plt.show()
