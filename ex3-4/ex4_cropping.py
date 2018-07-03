import cv2 as cv


def croppingImg(image, center_point,size):
    
    ptX,ptY = center_point
    sizeX,sizeY = size
    
    img = image[ptX-(sizeX%2):ptX-(sizeX%2-sizeX),ptY-(sizeY%2):ptY-(sizeY%2-sizeY)]
    
    return img


if __name__=='__main__':

    img = cv.imread('image.jpeg')
    pt = tuple(int(x.strip()) for x in (input("center point :").split(',')))
    size = tuple(int(x.strip()) for x in (input("size : ").split(',')))
    
    crp = croppingImg(img,pt,size)
    print(crp.shape)

    cv.imshow('cropped image',crp)
    cv.waitKey(0)
    cv.destroyAllWindows()
