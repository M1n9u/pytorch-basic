from glob import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

filenames = glob("data/T91/*.png")
original_images = []
hmin = 9999
wmin = 9999
for filename in filenames:
    if os.path.exists(filename):
        img = cv2.imread(filename)
        if img is not None:
            imsize = img.shape
            hmin, wmin = min(hmin, imsize[0]), min(wmin, imsize[1])
            original_images.append(img)

cropped_images = []
for img in iter(original_images):
    height, width = img.shape[:2]
    h_center, w_center = height//2, width//2
    h_start, w_start = h_center - (hmin//2), w_center - (wmin//2)
    cropped_images.append(img[h_start:h_start + hmin, w_start:w_start + wmin, :])
images = np.array(cropped_images)
print(images.shape)