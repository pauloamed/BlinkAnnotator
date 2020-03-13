from pynput import keyboard
import cv2
import os
import argparse
import time
import random

from tqdm import tqdm


ORIG_DUR = 1 / 30
TIMES_SLOWER = 3
MOD_CUR = ORIG_DUR * TIMES_SLOWER
FRAMES_WINDOW_SIZE = max(int(0.5 / MOD_CUR), 3)

print("Frame duration: " + str(MOD_CUR))
print("Frames window size: " + str(FRAMES_WINDOW_SIZE))

####################################################################################################


spacebarPressed = False

def onPress(key):
    global spacebarPressed
    try:
        if key == keyboard.Key.space: # space
            spacebarPressed = True
    except:
        pass

def onRelease(key):
    global spacebarPressed
    try:
        if key == keyboard.Key.space: # space
            spacebarPressed = False
    except:
        pass

####################################################################################################

def getDissonantIndexes(images, normal, framesDir):
    dissonantIndexes = []
    totalFrames = len(images)

    print("Hit SPACE for \'{}\'".format(normal.upper()))
    print("Press ENTER to continue")
    input()
    for classIndex, origIndex in tqdm(enumerate(images), total=len(images)):
        frame = cv2.imread(os.path.join(framesDir, "{}.jpg".format(origIndex)), cv2.IMREAD_COLOR)
        cv2.imshow("Frame", frame)

        time.sleep(MOD_CUR)

        if spacebarPressed:
            dissonantIndexes.append(classIndex)

        cv2.waitKey(1)

    finalDissonantIndexes = set()
    for classIndex in dissonantIndexes:
        left = max(0, classIndex - (FRAMES_WINDOW_SIZE))
        right = min(totalFrames, classIndex + 3)
        finalDissonantIndexes.update([images[i] for i in range(left, right)])

    return list(finalDissonantIndexes)


def manualClassify(maybeWrong, framesDir):
    certainBlink, certainNotBlink = [], []
    for cont, i in enumerate(maybeWrong):
        frame = cv2.imread(os.path.join(framesDir, "{}.jpg".format(i)), cv2.IMREAD_COLOR)
        cv2.imshow("Frame", frame)

        while True:
            print("({}/{}) Are the eyes open or closed? Type O for open and C for closed".format(cont + 1, len(maybeWrong)))
            annotation = input()
            if annotation == 'O':
                certainNotBlink.append(i)
                break
            elif annotation == 'C':
                certainBlink.append(i)
                break
            else:
                print("Wrong input!")

        cv2.waitKey(1)

    return certainBlink, certainNotBlink


def main(classes, framesDir):
    output = [None for _ in range(len(classes))]


    blinkImages = [i for i, x in enumerate(classes) if x == "blink"]
    notBlinkImages = [i for i, x in enumerate(classes) if x == "not blink"]

    for i in blinkImages: output[i] = True
    for i in notBlinkImages: output[i] = False

    maybeWrongBlinkImages = getDissonantIndexes(blinkImages, "open", framesDir)

    print("{} images selected".format(len(maybeWrongBlinkImages)))
    print("Press ENTER to continue")
    input()

    maybeWrongNotBlinkImages = getDissonantIndexes(notBlinkImages, "closed", framesDir)
    print("{} images selected".format(len(maybeWrongNotBlinkImages)))

    print("Press ENTER to continue")
    input()

    maybeWrong = maybeWrongBlinkImages + maybeWrongNotBlinkImages
    random.shuffle(maybeWrong)

    certainBlink, certainNotBlink = manualClassify(maybeWrong, framesDir)

    for i in certainBlink: output[i] = True
    for i in certainNotBlink: output[i] = False

    return output


####################################################################################################

def readCSV(inputPath):
    numEntries, originalClasses, frameStarts = None, [], []
    with open(inputPath) as file:
        lines = list(file.readlines())
        header = lines[0].split(';')

        frameStartIndex = header.index('frame_start')
        isBlinkingIndex = header.index('is_blinking')

        lines = lines[1:]
        numEntries = len(lines)

        for line in lines:
            line = line.split(';')
            frameStarts.append(line[frameStartIndex].strip())
            originalClasses.append(line[isBlinkingIndex].strip())

    return numEntries, originalClasses, frameStarts

####################################################################################################

listener = keyboard.Listener(on_press=onPress, on_release=onRelease)
listener.start()

####################################################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-rp", "--inputCSV", required=True, help="Caminho para arquivo CSV de entrada")
ap.add_argument("-fd", "--framesDir", required=True, help="Caminho para pasta onde ficarão salvas as imagens")
ap.add_argument("-op", "--outputPath", required=True, help="Caminho para arquivo CSV de saída")
args = vars(ap.parse_args())

####################################################################################################

numEntries, originalClasses, frameStarts = readCSV(args['inputCSV'])

####################################################################################################

correctedClasses = main(originalClasses, args['framesDir'])
with open(args['outputPath'], "w") as f:
    f.write("index;frame_start;is_blinking;")
    f.write('\n')

    for i in range(numEntries):
        values = [str(i), frameStarts[i], "blink" if correctedClasses[i] else "not blink"]
        f.write(";".join([x for x in values]))
        f.write('\n')
