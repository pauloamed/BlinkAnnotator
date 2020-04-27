import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-od", "--outputsDir", required = True)
parser.add_argument("-of", "--outputFile", required = True)
args = parser.parse_args()

with open(args.outputFile, "w") as out:
    for path in os.listdir(args.outputsDir):
        filePath = os.path.join(args.outputsDir, path)
        with open(filePath, "r") as f:
            for line in f.readlines():
                out.write(line)
