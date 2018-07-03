import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpeg')

print(img.shape,img.size)

r,g,b = cv2.split(img)
rgb = np.vstack((r,g,b))

img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(img_lab)
lab = np.vstack((l,a,b))

cv2.imshow('ironman_rgb',rgb)
cv2.imshow('ironman_lab',lab)
cv2.waitKey(0)
cv2.destroyAllWindows()
