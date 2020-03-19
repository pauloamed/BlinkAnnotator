import os

datasetPath = 'out'

openEyesFiles = []
closedEyesFiles = []

for x in os.listdir(datasetPath):
    y = x.split('_')
    if 'closed' in y:
        closedEyesFiles.append(x)
    else:
        openEyesFiles.append(x)

for i, x in enumerate(openEyesFiles):
    x = os.path.join(datasetPath, x)
    y = os.path.join(datasetPath, "open_{}.jpg".format(i))
    os.rename(x, y)

for i, x in enumerate(closedEyesFiles):
    x = os.path.join(datasetPath, x)
    y = os.path.join(datasetPath, "closed_{}.jpg".format(i))
    os.rename(x, y)


print("Exiting...")
