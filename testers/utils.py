from blink_utils.models.face_detectors import *
from blink_utils.models.classifiers import *
from blink_utils.models.roi_extractors import *

import os
import cv2
from tqdm import tqdm

def getStatsPerSec(currentSec, lastSec, itersSec):
    output = None

    # se virou o segundo (comparando o segundo atual com da iteracao passada)
    if currentSec != lastSec:
        # log da qntd de iteracoes do ultimo segundo
        output = itersSec
        lastSec = currentSec # atualizando o segundo atual
        itersSec = 0 # zerando a contagem de iteracoes do ultimo (agr atual) segundo

    # se houve face detectada, adicionar na contagem de faces detectadas pro segundo
    itersSec += 1 # incrementar a qtnd de iteracoes pro segundo

    return output, currentSec, lastSec, itersSec


def getModelClass(arg):
    try:
        model = eval(arg)
        return model
    except:
        print("erro", arg)
        exit(1)


def loadModels(args, filesPath):

    print(args)
    faceDetector = getModelClass(args.face_detector)(filesPath)

    roiExtractor = None
    if args.roi_extractor:
        try:
            roiExtractor = getModelClass(args.roi_extractor)(filesPath)
        except:
            roiExtractor = getModelClass(args.roi_extractor)()

    try:
        classifier = getModelClass(args.classifier)(filesPath)
    except:
        classifier = getModelClass(args.classifier)()

    return faceDetector, roiExtractor, classifier






def readCSV(inputPath):
    numEntries, originalClasses, frameStarts = None, [], []
    with open(inputPath) as file:
        lines = list(file.readlines())
        header = lines[0].split(';')

        frameStartIndex = header.index('frame_start')
        isBlinkingIndex = header.index('eyes_status')

        lines = lines[1:]
        numEntries = len(lines)

        for line in lines:
            line = line.split(';')
            frameStarts.append(line[frameStartIndex].strip())
            originalClasses.append(line[isBlinkingIndex].strip())

    return numEntries, originalClasses, frameStarts


def calcMetrics(TruePositives, FalsePositives, TrueNegatives, FalseNegatives):
    numEntries = TruePositives + FalsePositives + TrueNegatives + FalseNegatives

    accuracy = ((TruePositives + TrueNegatives) / numEntries)

    precision = 0
    if (TruePositives + FalsePositives) > 0:
        precision = (TruePositives / (TruePositives + FalsePositives))

    recall = 0
    if (TruePositives + FalseNegatives) > 0:
        recall = (TruePositives / (TruePositives + FalseNegatives))

    f1Score = 2 * ((precision * recall) / (precision + recall))

    return accuracy, precision, recall, f1Score


def testVideo(faceDetector, roiExtractor, classifier, imagesDir, imagesAnnotation, showFrames):

    numEntries, originalClasses, frameStarts = readCSV(imagesAnnotation)

    TruePositives = 0
    FalsePositives = 0
    TrueNegatives = 0
    FalseNegatives = 0

    for i in tqdm(range(numEntries)):
        frame = cv2.imread(os.path.join(imagesDir, "{}.jpg".format(i)), cv2.IMREAD_COLOR)

        facePoints = faceDetector(frame)
        faceFrame = frame[facePoints['y1']:facePoints['y2'], facePoints['x1']:facePoints['x2']]
        frame = faceFrame

        if roiExtractor:
            eyesPointsFromScales = roiExtractor(frame)
            eyesFrames = [frame[eyesPoints['y1']:eyesPoints['y2'], eyesPoints['x1']:eyesPoints['x2']]
                            for eyesPoints in eyesPointsFromScales]
            frame = eyesFrames[0]

        output = classifier(frame)
        detectorOutput = "closed" if output else "open"

        if originalClasses[i] == 'closed':
            if detectorOutput == 'closed':
                TruePositives += 1
            else:
                FalseNegatives += 1
        else:
            if detectorOutput == 'closed':
                FalsePositives += 1
            else:
                TrueNegatives += 1

        if showFrames == True:
            cv2.waitKey(20)


    outStr = ("Results on {}\n".format(imagesDir))
    outStr += ("\n\tTrue Positives: {}, False Positives: {}, False Negatives: {}, True Negatives: {}\n\n".format(
        TruePositives, FalsePositives, FalseNegatives, TrueNegatives
    ))

    accuracy, precision, recall, f1Score = calcMetrics(TruePositives, FalsePositives, TrueNegatives, FalseNegatives)

    outStr += ("\tAccuracy: %f\n\n" % accuracy)

    outStr += ("\tDos que o detector falou que eram fechados, qual porcentagem realmente era\n")
    outStr += ("\tPrecision: %f\n\n" % precision)

    outStr += ("\tDos que o detector precisava falar que eram fechados, qual porcentagem ele falou\n")
    outStr += ("\tRecall: %f\n\n" % recall)

    outStr += ("Eh melhor que precisao > recall. O usuario pode piscar de novo, mas o erro de uma falsa piscada é custoso\n")

    outStr += ("\tF1 Score: %f\n\n" % f1Score)

    return outStr
