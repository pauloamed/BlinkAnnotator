import os
import argparse

def renameFiles(pathsList, name):
    for i, x in enumerate(pathsList):
        y = os.path.join(datasetPath, name + "_{}.jpg".format(i))
        os.rename(x, y)

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--datasetPath", required=True, help="")
args = ap.parse_args()

datasetPath = args.datasetPath

openEyesFiles = []
closedEyesFiles = []

for x in os.listdir(datasetPath):
    y = x.split('_')

    newName = os.path.join(datasetPath, "~" + x)
    x = os.path.join(datasetPath, x)
    os.rename(x, newName)

    if 'closed' in y:
        closedEyesFiles.append(newName)
    else:
        openEyesFiles.append(newName)


renameFiles(openEyesFiles, "open")
renameFiles(closedEyesFiles, "closed")


print("Exiting...")
