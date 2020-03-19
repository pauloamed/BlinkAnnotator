from . import DlibEyesExtractor


def rescaleRectangle(x, y, w, h, scale):
    horScale, verScale = scale

    verticalCenter = y + h // 2
    horizontalCenter = x + w // 2

    w = int(w * horScale)
    h = int(h * verScale)

    y = verticalCenter - h // 2
    x = horizontalCenter - w // 2

    return x, y, w, h



class RectangleEyesExtractor():
    def __init__(self):

        self.landmarksPoints = DlibEyesExtractor('modelFiles', scales=[(1.1, 1.5)])

        self.x1Scale = .1
        self.x2Scale = .9
        self.y1Scale = .2
        self.y2Scale = .5

        self.updateRound = 300
        self.counterUpdate = 0


    def maybeUpdateRectangle(self, frame):
        if self.landmarksPoints is None:
            return

        frameWidth, frameHeight = frame.shape[1::-1]

        if self.counterUpdate == 0:
            eyePoints = self.landmarksPoints.extract(frame)[0]

            self.x1Scale = eyePoints['x1'] / frameWidth
            self.x2Scale = eyePoints['x2'] / frameWidth
            self.y1Scale = eyePoints['y1'] / frameHeight
            self.y2Scale = eyePoints['y2'] / frameHeight


        self.counterUpdate = (self.counterUpdate + 1) % self.updateRound


    def extract(self, frame):
        self.maybeUpdateRectangle(frame)

        frameWidth, frameHeight = frame.shape[1::-1]
        pointsDict = {
            'x1': int(frameWidth *  self.x1Scale),
            'x2': int(frameWidth *  self.x2Scale),
            'y1': int(frameHeight * self.y1Scale),
            'y2': int(frameHeight * self.y2Scale),
        }

        return pointsDict
