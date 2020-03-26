import os
import argparse
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dirPath", required=True, help="")
ap.add_argument("-op", "--outputPath", required=True, help="")
args = ap.parse_args()

if not os.path.exists(args.outputPath):
    os.mkdir(args.outputPath)


for subdirPath in os.listdir(args.dirPath):
    fullSubdirPath = os.path.join(args.dirPath, subdirPath)

    if not os.path.isdir(fullSubdirPath):
        continue

    for file in os.listdir(fullSubdirPath):
        newFileName = subdirPath + "_" + file
        src = os.path.join(fullSubdirPath, file)
        dst = os.path.join(args.outputPath, newFileName)
        shutil.copyfile(src, dst)
