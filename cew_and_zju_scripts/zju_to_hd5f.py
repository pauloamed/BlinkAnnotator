import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import threading
import h5py
import time

def fooLoadImages(files, desc, loadedFiles):
    print("Starting thread {} for loading images".format(desc))
    loadedFiles[0] = [cv2.imread(filePath, cv2.IMREAD_COLOR) for filePath in files]
    print("Finishing thread {} for loading images".format(desc))


def loadImagesFilesThreading(files, numThreads, maxPerThread = 500):
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


def loadImagesFromDir(dirPath, numThreads):
    files = [os.path.join(dirPath, item) for item in tqdm(os.listdir(dirPath), desc="Loading " + dirPath) if item.split('.')[-1] == 'jpg']
    images = loadImagesFilesThreading(files, numThreads)
    return images


def loadDataset(openEyesDir, closedEyesDir, numThreads = 4):
    openEyesImages = loadImagesFromDir(openEyesDir, numThreads)
    closedEyesImages = loadImagesFromDir(closedEyesDir, numThreads)

    return np.asarray(openEyesImages), np.asarray(closedEyesImages)


ap = argparse.ArgumentParser()
ap.add_argument("--openEyesDirTrain", required=True)
ap.add_argument("--closedEyesDirTrain", required=True)
ap.add_argument("--openEyesDirTest", required=True)
ap.add_argument("--closedEyesDirTest", required=True)
ap.add_argument("--numThreads", required = True)
ap.add_argument("--datasetHDF5Path", required = True)

args = ap.parse_args()

openEyesImagesTrain, \
closedEyesImagesTrain = loadDataset(args.openEyesDirTrain, args.closedEyesDirTrain, int(args.numThreads))

openEyesImagesTest, \
closedEyesImagesTest = loadDataset(args.openEyesDirTest, args.closedEyesDirTest, int(args.numThreads))

with h5py.File(args.datasetHDF5Path, "w") as out:
    openDatasetTrain = out.create_dataset("train_open", openEyesImagesTrain.shape,dtype='u1')
    closedDatasetTrain = out.create_dataset("train_closed", closedEyesImagesTrain.shape,dtype='u1')
    openDatasetTest = out.create_dataset("test_open", openEyesImagesTest.shape,dtype='u1')
    closedDatasetTest = out.create_dataset("test_closed", closedEyesImagesTest.shape,dtype='u1')

    openDatasetTrain[...] = openEyesImagesTrain
    closedDatasetTrain[...] = closedEyesImagesTrain
    openDatasetTest[...] = openEyesImagesTest
    closedDatasetTest[...] = closedEyesImagesTest
