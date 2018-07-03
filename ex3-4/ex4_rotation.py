import cv2 as cv
import numpy as np
import math

def rotateImg(image,angle):
    rows,cols = image.shape
    M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1) #(rotating_center,angle,scale)

    r = np.deg2rad(angle)
    new_rows, new_cols = (int(abs(rows*np.cos(r))+abs(cols*np.sin(r))),int(abs(cols*np.cos(r))+abs(rows*np.sin(r))))

    M[0,2] += (new_cols-cols)/2
    M[1,2] += (new_rows-rows)/2
    dst = cv.warpAffine(image,M,(new_cols,new_rows)) #(inputArray_src,rotationMat,size)

    return dst


if __name__=='__main__':

    img = cv.imread('image.jpeg',0)

    angle = input('Enter the angle of rotation : ')
    
    dst = rotateImg(img,float(angle))

    cv.imshow('rotation',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
