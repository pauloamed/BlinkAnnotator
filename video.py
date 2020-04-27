import cv2
import numpy as np
import glob

img_array = []
imgs = []
for filename in glob.glob('./renamed_dataset/teste/guiSemOculos/*'):
    imgs.append((filename, int((filename.split('/')[-1].split('_')[0]))))

imgs = sorted(imgs, key= lambda x : x[-1])
for filename, _ in imgs:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(500):
    out.write(img_array[i])
out.release()
