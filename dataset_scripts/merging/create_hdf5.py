import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import threading
import h5py
import time


def saveFilesByChunking(files, numThreads, maxPerThread = 500):
    numFiles = len(files)

    splits = [int(i * maxPerThread) for i in range(int(numFiles/maxPerThread))]

    threads = []
    for i in range(len(splits)):
        loadedFiles = [i]
        if i == len(splits) - 1:
            filesOfInterest = files[splits[-1]:]
        else:
            filesOfInterest = files[splits[i]:splits[i + 1]]
        t = threading.Thread(target=fooLoadImages, args=(filesOfInterest, "Thread #{}, numFiles: {}".format(i, len(filesOfInterest)), loadedFiles))

        while threading.active_count() > numThreads:
            time.sleep(5)

        t.start()
        threads.append((t, loadedFiles))

    images = []
    for x in threads:
        x[0].join()
        images += x[1][0]

    return images


def appendToDataset(files, h5File, datasetName, chunkSize):
    numFiles = len(files)
    splits = [int(i * chunkSize) for i in range(int(numFiles/chunkSize))]

    for i in range(len(splits)):
        if i == len(splits) - 1:
            l = splits[-1]
            r = len(files)
        else:
            l = splits[i]
            r = splits[i + 1]

        print(">> Started processing from {} to {}".format(l, r))

        images = [cv2.imread(file, cv2.IMREAD_COLOR) for file in files[l:r]]

        with h5py.File(args.datasetHDF5Path, "a") as out:
            out["open"][l:r, ...] = images

        print("<< Finished processing from {} to {}".format(l, r))

def loadDataset(imagesDir):
    files = [os.path.join(imagesDir, item) for item in os.listdir(imagesDir) if item.split('.')[-1] == 'jpg']
    openFiles = [file for file in files if "open" in file]
    closedFiles = [file for file in files if "closed" in file]

    return openFiles, openFiles


ap = argparse.ArgumentParser()
ap.add_argument("--imagesDir", required=True)
ap.add_argument("--datasetHDF5Path", required = True)

args = ap.parse_args()

openFiles, closedFiles = loadDataset(args.imagesDir)

shape = cv2.imread(openFiles[0], cv2.IMREAD_COLOR).shape

with h5py.File(args.datasetHDF5Path, "w") as out:
    openDataset = out.create_dataset("open", (len(openFiles), *shape),dtype='u1')
    closedDataset = out.create_dataset("closed", (len(closedFiles), *shape),dtype='u1')


appendToDataset(openFiles, args.datasetHDF5Path, "open", 500)
appendToDataset(closedFiles, args.datasetHDF5Path, "closed", 500)

test = None
with h5py.File(args.datasetHDF5Path, "r") as f:
    test = out["open"][0,...]
cv2.imshow("a", test)
