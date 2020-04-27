import argparse
import datetime
import subprocess
import psutil
import time
import os
import threading
from tqdm import tqdm
from random import shuffle

FNULL = open(os.devnull, 'w')
facesPath = None
videosPath = None


def maybeCreateDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def getSize():
    tot = 0
    for f in os.listdir(facesPath):
        fp = os.path.join(facesPath, f)
        tot += os.path.getsize(fp)
    for f in os.listdir(videosPath):
        fp = os.path.join(videosPath, f)
        tot += os.path.getsize(fp)
    return tot


def runCommands(commands, idx):
    shuffle(commands)
    i = 0
    while i < len(commands):
        size = getSize()
        if size > 2 * (1024 ** 3):
            time.sleep(10)
            continue

        print("{}: ({}/{}), size = {}".format(idx, i + 1, len(commands), size))
        subprocess.run(commands[i], stdout=FNULL, stderr=subprocess.STDOUT)
        i += 1


parser = argparse.ArgumentParser()
parser.add_argument("-op", "--outputPath", required = True)
parser.add_argument("-t", "--numThreads", required = True)
args = parser.parse_args()


outputPath = args.outputPath
videosPath = os.path.join(outputPath, "videos")
facesPath = os.path.join(outputPath, "faces")
maybeCreateDir(outputPath)
maybeCreateDir(videosPath)
maybeCreateDir(facesPath)

alreadyThere = set()
for x in os.listdir(facesPath):
    alreadyThere.add(x.rsplit('_', 1)[0])


commands = []
with open("avspeech_train.csv", "r") as f:
    for line in tqdm(f.readlines()):
        line = line.split(',')
        videoId = line[0]

        if videoId in alreadyThere:
            continue

        start = float(line[1])
        end = min(float(line[2]), start + 60)

        start = str(datetime.timedelta(seconds=start))
        end = str(datetime.timedelta(seconds=end))

        command = [ "youtube-dl",
                    "https://www.youtube.com/watch?v={}".format(videoId),
                    "--external-downloader",
                    "ffmpeg",
                    "-f",
                    "webm",
                    "--external-downloader-args",
                    "-ss {} -to {}".format(start, end),
                    "-o",
                    os.path.join(videosPath, "{}.webm".format(videoId))
        ]

        commands.append(command)


numThreads = int(args.numThreads)
chunkSize = len(commands) // numThreads
splits = [[chunkSize * i, chunkSize * (i + 1)] for i in range(numThreads)]
splits[-1][-1] = len(commands)

for i, (l, r) in enumerate(splits):
    t = threading.Thread(target=runCommands, args=(commands[l:r], i))
    t.start()
