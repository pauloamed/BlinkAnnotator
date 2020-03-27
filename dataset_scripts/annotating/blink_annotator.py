from pynput import keyboard
import cv2
import os
import argparse
import time
import random

from tqdm import tqdm

from utils import countdown

####################################################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--inputPath", required=True, help="Caminho para arquivo CSV de entrada")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-op", "--outputPath", required=True, help="Caminho para arquivo CSV de saída")
ap.add_argument("-ts", "--timesSlower", required=False, help="Quantas vezes mais lento que a execução normal", default=3)
ap.add_argument("-st", "--seqThreshold", required=False, help="Threshold de sequencias", default=3)
args = vars(ap.parse_args())

####################################################################################################

ORIG_DUR = 1 / 30
TIMES_SLOWER = int(args['timesSlower'])
MOD_DUR = ORIG_DUR * TIMES_SLOWER
FRAMES_WINDOW_SIZE = max(int(1 / MOD_DUR), 3)

SEQ_THRESHOLD = int(args['seqThreshold'])

####################################################################################################


spacebarPressed = False
rPressed = False
timesPressed = 0

def onPress(key):
    global spacebarPressed, rPressed
    try:
        if key == keyboard.Key.space: # space
            spacebarPressed = True
        if key.char == "r": # r
            rPressed = True
    except:
        pass

def onRelease(key):
    global spacebarPressed, rPressed
    try:
        if key == keyboard.Key.space: # space
            spacebarPressed = False
            timesPressed += 1
        if key.char == "r": # r
            rPressed = False
    except:
        pass

####################################################################################################

def getDissonantIndexes(images, normal, framesDir):
    totalFrames = len(images)

    resetRequest = True
    while resetRequest:
        resetRequest = False

        dissonantIndexes = []
        print("Hit SPACE for \'{}\'".format(normal.upper()))
        print("Press ENTER to continue")
        input()
        countdown(5)
        for classIndex, origIndex in tqdm(enumerate(images), total=len(images)):
            addFrame = False

            frame = cv2.imread(os.path.join(framesDir, "{}.jpg".format(origIndex)), cv2.IMREAD_COLOR)
            if frame is None:
                print("Error while openning {}.jpg".format(origIndex))
                print("Exiting...")
                exit()

            cv2.imshow("", frame)

            if rPressed:
                print("Restarting step...")
                resetRequest = True
                cv2.destroyAllWindows()
                break

            timesPressedBefore = timesPressed
            time.sleep(MOD_DUR)

            if spacebarPressed:
                addFrame = True

            timesPressedAfter = timesPressed
            if timesPressedAfter > timesPressedBefore:
                addFrame = True

            if addFrame:
                dissonantIndexes.append(classIndex)

            cv2.waitKey(1)

            if classIndex == len(images) - 1:
                while True:
                    print("Would you like to reset this step? Y for Yes and N for No")
                    ans = input()
                    if ans == 'Y':
                        print("Restarting step...")
                        resetRequest = True
                        cv2.destroyAllWindows()
                        break
                    elif ans == 'N':
                        print("Confirmed!")
                        break
                    else:
                        print("Wrong input!")

    finalDissonantIndexes = set()
    for classIndex in dissonantIndexes:
        left = max(0, classIndex - (FRAMES_WINDOW_SIZE))
        right = min(totalFrames, classIndex + 3)
        finalDissonantIndexes.update([images[i] for i in range(left, right)])

    return list(finalDissonantIndexes)


def resetListSteps(l, i):
    while len(l) > 0 and l[-1] >= i:
        l.pop(-1)


def manualClassify(maybeWrong, framesDir):
    certainBlink, certainNotBlink = [], []


    cont = 0
    while cont < len(maybeWrong):
        resetRequest = False
        i = maybeWrong[cont]

        frame = cv2.imread(os.path.join(framesDir, "{}.jpg".format(i)), cv2.IMREAD_COLOR)
        cv2.putText(frame, str(i), (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


        while True:
            print("({}/{}) Frame {}: Are the eyes open or closed? Type O for open, C for closed and r for reseting steps".format(cont + 1, len(maybeWrong), i))
            cv2.imshow("", frame)
            cv2.waitKey(150)

            annotation = input()
            if annotation == 'O':
                certainNotBlink.append(cont)
                break
            elif annotation == 'C':
                certainBlink.append(cont)
                break
            elif annotation == 'r':
                cv2.destroyAllWindows()
                print("To which step would you like to go back?")

                try:
                    newStep = int(input()) - 1
                except:
                    newStep = cont - 1

                newStep = max(newStep, 0)
                newStep = min(newStep, cont)

                resetListSteps(certainBlink, newStep)
                resetListSteps(certainNotBlink, newStep)

                cont = newStep
                resetRequest = True

                break
            else:
                print("Wrong input!")

        if resetRequest:
            continue

        cont += 1

    for i in range(len(certainBlink)):
        certainBlink[i] = maybeWrong[certainBlink[i]]

    for i in range(len(certainNotBlink)):
        certainNotBlink[i] = maybeWrong[certainNotBlink[i]]


    return certainBlink, certainNotBlink


def checkForShortSequences(output):
    seqSize = 0
    seqStatus = output[0]

    badSequences = []

    i = 0
    while i <= len(output):
        if i < len(output) and output[i] == seqStatus:
            seqSize += 1
        else:
            if seqSize <= SEQ_THRESHOLD:
                seqList = list(range(i - seqSize, i))
                if len(badSequences) >= 1 and badSequences[-1][-1] == seqList[0] - 1:
                    badSequences[-1] += seqList
                else:
                    badSequences.append(seqList)
            seqSize = 1

            if i < len(output):
                seqStatus = output[i]

        i += 1

    return badSequences


def manualClassifyShortSeqs(shortSequences, framesDir):
    certainBlink, certainNotBlink = [], []
    for i, seq in enumerate(shortSequences):
        print("({}/{}) Correcting the sequence {}:".format(i+1, len(shortSequences), i+1))

        print("Reproducing sequence...")

        playSequence = True
        while playSequence:
            for j in seq:
                frame = cv2.imread(os.path.join(framesDir, "{}.jpg".format(j)), cv2.IMREAD_COLOR)
                cv2.putText(frame, str(j), (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("", frame)
                cv2.waitKey(100)

            while True:
                print("Play sequence again? Type Y for yes and N for no")
                ans = input()
                if ans == 'N':
                    playSequence = False
                    break
                elif ans == 'Y':
                    playSequence = True
                    break
                else:
                    print("Wrong input")

        certainBlinkSeq, certainNotBlinkSeq = manualClassify(seq, framesDir)
        certainBlink += certainBlinkSeq
        certainNotBlink += certainNotBlinkSeq

    return certainBlink, certainNotBlink


def main(classes, framesDir):
    output = [None for _ in range(len(classes))]


    blinkImages = [i for i, x in enumerate(classes) if x == "closed"]
    notBlinkImages = [i for i, x in enumerate(classes) if x == "open"]

    for i in blinkImages: output[i] = True
    for i in notBlinkImages: output[i] = False

    print("============================================================")
    print("============================================================")
    print("=============          FIRST PHASE           ===============")
    print("============================================================")
    print("============================================================")
    print("============================================================")
    print("Frame duration: " + str(MOD_DUR))
    print("Frames window size: " + str(FRAMES_WINDOW_SIZE))
    print("============================================================\n")
    print("Press ENTER to continue:")
    input()

    maybeWrongBlinkImages = getDissonantIndexes(blinkImages, "open", framesDir)
    print("{} images selected".format(len(maybeWrongBlinkImages)))

    cv2.destroyAllWindows()

    maybeWrongNotBlinkImages = getDissonantIndexes(notBlinkImages, "closed", framesDir)
    print("{} images selected".format(len(maybeWrongNotBlinkImages)))

    cv2.destroyAllWindows()

    print("============================================================")
    print("============================================================")
    print("============          SECOND PHASE           ===============")
    print("============================================================")
    print("============================================================")
    print("Press ENTER to continue:")
    input()

    maybeWrong = maybeWrongBlinkImages + maybeWrongNotBlinkImages
    random.shuffle(maybeWrong)

    certainBlink, certainNotBlink = manualClassify(maybeWrong, framesDir)

    for i in certainBlink: output[i] = True
    for i in certainNotBlink: output[i] = False

    print("============================================================")
    print("============================================================")
    print("=============          THIRD PHASE           ===============")
    print("============================================================")
    print("============================================================")
    print("Press ENTER to continue:")
    input()

    shortSequences = checkForShortSequences(output)

    certainBlink, certainNotBlink = manualClassifyShortSeqs(shortSequences, framesDir)

    for i in certainBlink: output[i] = True
    for i in certainNotBlink: output[i] = False

    print("Exiting...")

    return output


####################################################################################################

def readCSV(inputPath):
    numEntries, originalClasses, frameStarts = None, [], []
    with open(inputPath) as file:
        lines = list(file.readlines())
        header = lines[0].split(';')

        frameStartIndex = header.index('frame_start')
        eyesStatusIndex = header.index('eyes_status')

        lines = lines[1:]
        numEntries = len(lines)

        for line in lines:
            line = line.split(';')
            frameStarts.append(line[frameStartIndex].strip())
            originalClasses.append(line[eyesStatusIndex].strip())

    return numEntries, originalClasses, frameStarts

####################################################################################################

listener = keyboard.Listener(on_press=onPress, on_release=onRelease)
listener.start()

####################################################################################################

numEntries, originalClasses, frameStarts = readCSV(args['inputPath'])

####################################################################################################

correctedClasses = main(originalClasses, args['framesDir'])
with open(args['outputPath'], "w") as f:
    f.write("index;frame_start;eyes_status;")
    f.write('\n')

    for i in range(numEntries):
        values = [str(i), frameStarts[i], "closed" if correctedClasses[i] else "open"]
        f.write(";".join([x for x in values]))
        f.write('\n')
