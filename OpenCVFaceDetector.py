import cv2
import random
from math import hypot


class OpenCVFaceDetector():
    def __init__(self, dataPath, cnnModel='tensorflow', cnnThreshold=0.9):

        if cnnModel == 'tensorflow':
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            modelLoader = cv2.dnn.readNetFromTensorflow
        elif cnnModel == 'caffe':
            modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "deploy.prototxt"
            modelLoader = cv2.dnn.readNetFromCaffe

        try:
            self.cnnNet = modelLoader(dataPath + modelFile, dataPath + configFile)
        except:
            print('Não foi possível carregar um dos seguintes arquivos:\n{}\n{}'.format(modelFile, configFile))
            exit(1)

        self.cnnThreshold = cnnThreshold


    def detect(self, frame):

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.cnnNet.setInput(blob) # blob sera a entrada da rede
        detections = self.cnnNet.forward() # deteccoes feitas na rede

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.cnnThreshold:
                face = {
                    'conf' : confidence,
                    'x1' : max(int(detections[0, 0, i, 3] * frame.shape[1]), 0),
                    'y1' : max(int(detections[0, 0, i, 4] * frame.shape[0]), 0),
                    'x2' : int(detections[0, 0, i, 5] * frame.shape[1]),
                    'y2' : int(detections[0, 0, i, 6] * frame.shape[0]),
                }
                return face
