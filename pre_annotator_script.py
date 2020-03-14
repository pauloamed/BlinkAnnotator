from os import listdir
from os.path import join
import subprocess
import time

datasetPath = 'dataset'
numProcessesLimit = 2

counter = dict()
for x in listdir(datasetPath):
    x = x.split('.')[0]
    if x in counter: counter[x] += 1
    else: counter[x] = 1

activeProcesses = []

counterItems = list(counter.items())
i = 0
while i < len(counterItems):
    x, count = counterItems[i]
    if count != 2:
        print("{} has counter not equals to 2".format(x))
        i += 1
    elif len(activeProcesses) < numProcessesLimit:
        framesDir = join(datasetPath, x)
        recordsPath = join(datasetPath, x + ".pickle")
        outputPath = join(datasetPath, x + ".csv")

        command = "python pre_annotator.py -fd {} -rp {} -op {}".format(framesDir, recordsPath, outputPath)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        activeProcesses.append((x, process))
        i += 1

    else:
        time.sleep(5)
        for x, process in activeProcesses:
            if process.poll() is None:
                print("Process {} not finished".format(x))
            else:
                print("Process {} finished".format(x))
                activeProcesses.remove((x, process))

while len(activeProcesses) != 0:
    for x, process in activeProcesses:
        if process.poll() is None:
            print("Process {} not finished".format(x))
        else:
            print("Process {} finished".format(x))
            activeProcesses.remove((x, process))



pritn("Exiting...")
