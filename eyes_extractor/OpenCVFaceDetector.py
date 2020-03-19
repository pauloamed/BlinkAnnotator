import cv2
import random
from math import hypot


class OpenCVFaceDetector():
    def __init__(self, dataPath, cnnModel='tensorflow', cnnThreshold=0.8):

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


    def getDetectedFacesAboveThreshold(self, frame):

        # nao tive tempo ainda de ler e entender tudo, fica de TODO pra reuniao
        # fonte: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
        # pelo o que eu entendi, blob eh um processamento feito na imagem. como isso acontece e porque eu nao sei ainda
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.cnnNet.setInput(blob) # blob sera a entrada da rede
        detections = self.cnnNet.forward() # deteccoes feitas na rede

        ## usar filter pra fazer isso, codigo baixo nivel por enquanto
        aboveThresholdFaces = []

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
                aboveThresholdFaces.append(face)

        return aboveThresholdFaces

    def facePunct(self, face, frameCenter):
        area = abs(face['x2'] - face['x1']) * abs(face['y2'] - face['y1'])
        faceCenter = ((face['x1'] + face['x2']) // 2, (face['y1'] + face['y2']) // 2)
        distCenters = hypot(frameCenter[0] - faceCenter[0], frameCenter[1] - faceCenter[1])
        confidence = face['conf']
        return area * 1/distCenters * confidence


    def detect(self, frame, markFrame=False):

        detectedFaces = self.getDetectedFacesAboveThreshold(frame)

        frameCenter = (frame.shape[0]//2, frame.shape[1]//2)

        try:
            bestFace = max(detectedFaces, key=lambda x : self.facePunct(x, frameCenter))
        except:
            return None

        if markFrame: # nao sei se vai dar certo, depende se frame for imutavel ou nao
            for face in detectedFaces:
                color = tuple(random.randint(0, 255) for _ in range(3))
                frame = cv2.rectangle(frame, (face['x1'], face['y1']),
                                             (face['x2'], face['y2']), color, 4)

        return bestFace
