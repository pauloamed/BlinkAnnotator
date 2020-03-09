import argparse
import pickle
import dlib
import cv2
import os


######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-rp", "--recordsPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-op", "--outputPath", required=True, help="Caminho para arquivo de saída")
args = vars(ap.parse_args())

############################## LANDMARKS PREDICTOR PREP ################################

dataPath = "modelFiles/"

################################# PICKLE/INPUT LOGIC ###################################

pickle_in = open(args["recordsPath"], "rb")
records = pickle.load(pickle_in)
pickle_in.close()

##################################### MAIN LOOP #############################################

with open(args['outputPath'], "w") as f:

    ## HEADER
    f.write("frame_start;is_blinking;avg")
    f.write('\n')

    ## FOR EACH SAVED RECORD/FRAME
    for i, record in enumerate(records):

        ## RETRIEVING FACE FRAME AND LANDMARKS
        faceFrame = cv2.imread(os.path.join(args['framesDir'], "{}.jpg".format(i)), cv2.IMREAD_COLOR)

        ## RETRIEVING VALUES TO BE WRITTEN ON CSV FILE
        isBlinking, frameStart = record

        values = ["Blink"] if isBlinking else ["Not blink"]

        width = faceFrame.shape[0]
        height = faceFrame.shape[1]

        top = int(height * .3)
        bottom = int(height * .7)
        left = int(height * .1)
        right = int(height * .9)

        faceFrame = faceFrame[top:bottom, left:right]
        faceFrame = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2GRAY)
        val = faceFrame.mean()
        values.append(val)

        # print(val)

        # cv2.imshow("a", faceFrame)
        # cv2.waitKey(0)



        ## WRITING
        f.write("%.3f;" % frameStart)
        f.write(";".join([str(x) for x in values]))
        f.write('\n')
