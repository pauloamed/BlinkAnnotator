import time
import argparse
import pickle
import os
import cv2
from math import floor
import threading
import random


AUDIO_PATH = os.path.abspath('beep.mp3')

######################################## ARGS ##########################################

ap = argparse.ArgumentParser()
ap.add_argument("-rp", "--recordsPath", required=True, help="Caminho para arquivo onde ficarão salvas as informacoes para cada frame")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-lib", "--audioLibrary", required=False)
args = ap.parse_args()

################################# AUDIO LIB SETTINGS ###################################

def pydubAudioPlayer(audioPath):
    from pydub import AudioSegment
    from pydub.playback import play

    audioFile = AudioSegment.from_mp3(audioPath)

    def playBeep():
        play(audioFile)

    return playBeep

def playsoundAudioPlayer(audioPath):
    from playsound import playsound

    def playBeep():
        playsound(audioPath)

    return playBeep

playBeep = None
if args.audioLibrary and args.audioLibrary == 'pydub':
    playBeep = pydubAudioPlayer(AUDIO_PATH)
else:
    playBeep = playsoundAudioPlayer(AUDIO_PATH)



######################### VIDEO CAPTURE + FACE DETECT PREP #############################

videoCapture = None
try:
    videoCapture = cv2.VideoCapture(0)
except:
    print('Não foi possível carregar o dispositivo de captura de vídeo')
    exit()

_, frame = videoCapture.read()

####################################### OUTPUT PREP ###################################

if not os.path.exists(args.framesDir):
    os.mkdir(args.framesDir)

##################################### AUX VARS INIT ##################################

def getNumbers():
    a = random.randint(10, 35)

    while True:
        b = random.randint(10, 35)
        if abs(b - a) <= 4:
            continue
        else:
            if a > b: return (b, a)
            else: return (a, b)

records = []
frameRate = -1
blinks = getNumbers()

print("Piscadas aos {} e {} segundos".format(*blinks))

firstBlinkPlayed = False
secondBlinkPlayed = False


##################################### THREADING ######################################

locks = [threading.Lock() for _ in (range(2))]
ready = threading.Event()

def playAudio():
    global locks
    ready.set()
    locks[0].acquire()
    playBeep()
    locks[1].acquire()
    playBeep()

blinkThread = threading.Thread(target=playAudio, args=())
locks[0].acquire()
locks[1].acquire()
blinkThread.start()

ready.wait()

##################################### MAIN LOOP ######################################

loopStart = time.time()

print("Press ENTER to continue")
input()
for i in range(3, 0, -1):
    print("Starting in {}...".format(i))
    time.sleep(1)

while (time.time() - loopStart) <= 40: ## EXECUTE LOOP FOR 60 SECS
    frameStart = time.time()
    relDuration = frameStart - loopStart


    if floor(relDuration) >= blinks[0] and not firstBlinkPlayed:
        firstBlinkPlayed = True
        locks[0].release()

    if floor(relDuration) >= blinks[1] and not secondBlinkPlayed:
        secondBlinkPlayed = True
        locks[1].release()




    ## FRAME CAPTURE AND FACE FRAME EXTRACTION
    _, frame = videoCapture.read()
    # frame = cv2.resize(frame, (320, 240))

    cv2.waitKey(1)
    ## SAVING OUTPUT

    cv2.imwrite(os.path.join(args.framesDir, "{}.jpg".format(len(records))), frame)
    records.append((relDuration))

    ## FEEDBACK LOGIC


    cv2.putText(frame, "FPS: {}".format(frameRate), (15, int(frame.shape[0] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(frame, "%.2f" % (time.time() - loopStart), (15, int(frame.shape[0] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('Cam', frame)

    ## FRAME RATE LOGIC

    frameEnd = time.time()
    frameDuration = frameEnd - frameStart
    frameRate = int(1 / frameDuration)


############################ PICKLE/OUTPUT LOGIC #############################


pickle_out = open(args.recordsPath, "wb")
pickle.dump(records, pickle_out)
pickle_out.close()
