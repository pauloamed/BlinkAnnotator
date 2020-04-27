import os
import argparse
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dirPath", required=True, help="")
ap.add_argument("-k", "--keys", required=True, help="")
args = ap.parse_args()

keys = args.keys.split('-')
keysFiles = [[] for key in keys]

for filePath in os.listdir(args.dirPath):
    for i, key in enumerate(keys):
        if key in filePath.split('_'):
            keysFiles[i].append(filePath)

for i, key in enumerate(keys):
    keyDir = os.path.join(args.dirPath, key)
    if not os.path.exists(keyDir):
        os.mkdir(keyDir)

    for filePath in keysFiles[i]:
        oldFilePath = os.path.join(args.dirPath, filePath)
        newFilePath = os.path.join(keyDir, filePath)

        shutil.move(oldFilePath, newFilePath)
