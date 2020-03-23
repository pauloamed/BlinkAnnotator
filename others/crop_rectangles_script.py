from os import listdir
from os.path import join
import subprocess
import time

datasetPath = 'dataset'
numProcessesLimit = 3
outputPath = 'out'
modelsPath = 'modelFiles'


items = []
for x in listdir(datasetPath):
    x = x.split('.')[0]
    if '_' in x:
        items.append(x)

activeProcesses = []


i = 0
while i < len(items):
    x = items[i]
    if len(activeProcesses) < numProcessesLimit:
        dir = x.split('_')[0]
        framesDir = join(datasetPath, dir)
        csvPath = join(datasetPath, x + ".csv")

        command = "python crop_rectangles.py -fd {} -op {} -mp {} -cp {}".format(framesDir, outputPath, modelsPath, csvPath)
        print("Starting " + command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        activeProcesses.append((x, process))
        i += 1
    else:
        time.sleep(5)
        for x, process in activeProcesses:
            out, err = process.communicate()
            print(out.decode("utf-8"))
            print(err.decode("utf-8"))
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

print("Exiting...")
