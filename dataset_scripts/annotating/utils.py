
def countdown(x, windowName = "", imsize = (300, 300)):
    import cv2
    import time
    import numpy as np

    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (100, 200)
    fontScale = 5
    fontColor = (255,255,255)
    thickness = 3

    for i in range(x, 0, -1):
        img = np.zeros(imsize)

        cv2.putText(img, str(i), pos, font, fontScale, fontColor, thickness)
        cv2.imshow(windowName, img)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()
