import dlib
import cv2
import os
import numpy as np


def rescaleRectangle(x, y, w, h, scale):
    horScale, verScale = scale

    verticalCenter = y + h // 2
    horizontalCenter = x + w // 2

    w = int(w * horScale)
    h = int(h * verScale)

    y = verticalCenter - h // 2
    x = horizontalCenter - w // 2

    return x, y, w, h


class DlibEyesExtractor():

    def __init__(self, dataPath, points = [],  scales = [(1.1, 1.5), (1.1, 2)]):
        predictorPath = "shape_predictor_68_face_landmarks.dat"
        self.scales = scales

        self.points = [36, 45, 21, 22, 29, 17, 26]

        try:
            self.predictor = dlib.shape_predictor(os.path.join(dataPath, predictorPath))
        except:
            print('Não foi possível carregar o seguinte arquivo:\n{}'.format(predictorPath))


    def extract(self, frame):
        landmarks = self.predictor(frame, dlib.rectangle(0, 0, frame.shape[1], frame.shape[0]))
        eyePoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.points])

        x, y, w, h = cv2.boundingRect(eyePoints)

        ret = []
        for scale in self.scales:
            rx, ry, rw, rh = rescaleRectangle(x, y, w, h, scale)

            rectCoords = {
            'x1' : max(rx, 0),
            'x2' : min(rx + rw, frame.shape[0]),
            'y1' : max(ry, 0),
            'y2' : min(ry + rh, frame.shape[1]),
            }

            ret.append(rectCoords)

        return ret
