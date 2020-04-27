import argparse
import os
from shutil import copyfile
from tqdm import tqdm


def loadAnnotationsDict(filePath):
    annots = dict()
    with open(filePath, "r") as f:
        for line in tqdm(f.readlines(), "Prearing annotations dict"):
            line = [x.strip() for x in line.split(';')]
            annots[line[0]] = [float(x) for x in line[1:5]]
    return annots


def maybeCreateDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def writeFiles(itemsList, originDir, outputDir):
    for item in tqdm(itemsList, desc="Writing into " + outputDir):
        src = os.path.join(originDir, item)
        dst = os.path.join(outputDir, item)
        copyfile(src, dst)

parser = argparse.ArgumentParser()
parser.add_argument("-od", "--annotationsFile", required = True)
parser.add_argument("-of", "--facesDir", required = True)
parser.add_argument("-dd", "--datasetDir", required=True)
args = parser.parse_args()

DATASET_DIR = args.datasetDir
OPEN_DIR = os.path.join(DATASET_DIR, "open")
CLOSED_DIR = os.path.join(DATASET_DIR, "closed")
MIN_DIFF = 0.99

maybeCreateDir(DATASET_DIR)
maybeCreateDir(OPEN_DIR)
maybeCreateDir(CLOSED_DIR)

annoDict = loadAnnotationsDict(args.annotationsFile)

openFiles = []
closedFiles = []
for path in tqdm(os.listdir(args.facesDir), desc="Extracting \"good\" samples"):
    try:
        openLeftProb, closedLeftProb, openRightProb, closedRightProb = annoDict[path]
    except:
        print(path + " not in annotations file!")
        continue

    leftMargin = abs(openLeftProb - closedLeftProb)
    rightMargin = abs(openRightProb - closedRightProb)

    if leftMargin < MIN_DIFF or rightMargin < MIN_DIFF:
        continue

    if closedLeftProb > openLeftProb and closedRightProb > openRightProb:
        closedFiles.append(path)
    elif openLeftProb > closedLeftProb and openRightProb > closedRightProb:
        openFiles.append(path)



print("Num of closed files: {}".format(len(closedFiles)))
print("Num of open files: {}".format(len(openFiles)))
print("Num of \"bad\" files: {}".format(len(os.listdir(args.facesDir)) - (len(openFiles) + len(closedFiles))))

writeFiles(closedFiles, args.facesDir, CLOSED_DIR)
writeFiles(openFiles, args.facesDir, OPEN_DIR)
