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
predictor = dlib.shape_predictor(os.path.join(dataPath, "shape_predictor_68_face_landmarks.dat"))

################################# PICKLE/INPUT LOGIC ###################################

pickle_in = open(args["recordsPath"], "rb")
records = pickle.load(pickle_in)
pickle_in.close()

##################################### MAIN LOOP #############################################

with open(args['outputPath'], "w") as f:

    ## HEADER
    f.write("frame_start;is_blinking;")
    f.write(";".join(["{}x;{}y".format(i, i) for i in range(36, 48)]))
    f.write('\n')

    ## FOR EACH SAVED RECORD/FRAME
    for i, record in enumerate(records):

        ## RETRIEVING FACE FRAME AND LANDMARKS
        faceFrame = cv2.imread(os.path.join(args['framesDir'], "{}.jpg".format(i)), cv2.IMREAD_COLOR)
        landmarks = predictor(faceFrame, dlib.rectangle(0, 0, faceFrame.shape[1], faceFrame.shape[0]))

        ## RETRIEVING VALUES TO BE WRITTEN ON CSV FILE
        isBlinking, frameStart = record
        values = [isBlinking]
        for i in range(36, 48):
            values.append(landmarks.part(i).x)
            values.append(landmarks.part(i).y)

        ## WRITING
        f.write("%.3f;" % frameStart)
        f.write(";".join([str(x) for x in values]))
        f.write('\n')
