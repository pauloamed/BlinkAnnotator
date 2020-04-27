# le arquivo out.txt
import argparse
import os
from shutil import copyfile

def maybeCreateDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def writeFiles(itemsList, originDir, outputDir):
    for item in itemsList:
        src = os.path.join(originDir, item)
        dst = os.path.join(outputDir, item)
        copyfile(src, dst)


ap = argparse.ArgumentParser()
ap.add_argument("--annotationsFile", required=True)
ap.add_argument("--datasetDir", required=True)
args = ap.parse_args()


notEye = 0
openEye = 0
closedEye = 0

closed = set()
opens = set()
bads = set()
with open(args.annotationsFile, "r") as f:
    for line in f.readlines():
        line = line.split(' ')
        filepath = line[0]
        probEyeAOpen = float(line[1])
        probEyeAClosed = float(line[2])
        probEyeBOpen = float(line[3])
        probEyeBClosed = float(line[4])

        badA = abs(probEyeAClosed - probEyeAOpen) < 0.7
        badB = abs(probEyeBClosed - probEyeBOpen) < 0.7

        bad = badA or badB

        if bad:
            bads.add(filepath)
            continue

        isClosed = probEyeAClosed > probEyeAOpen and probEyeBClosed > probEyeBOpen

        if isClosed:
            closed.add(filepath)
        else:
            opens.add(filepath)

badPaths = []
closedPaths = []
openPaths = []
for file in os.listdir(args.datasetDir):
    if file in bads:
        badPaths.append(file)
    elif file in closed:
        closedPaths.append(file)
    elif file in opens:
        openPaths.append(file)


maybeCreateDir(os.path.join(args.datasetDir, "bad"))
writeFiles(badPaths, args.datasetDir, os.path.join(args.datasetDir, "bad"))
maybeCreateDir(os.path.join(args.datasetDir, "open"))
writeFiles(openPaths, args.datasetDir, os.path.join(args.datasetDir, "open"))
maybeCreateDir(os.path.join(args.datasetDir, "closed"))
writeFiles(closedPaths, args.datasetDir, os.path.join(args.datasetDir, "closed"))
