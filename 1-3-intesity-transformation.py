import sys
import cv2 as cv
import numpy as np

def main():
    if len(sys.argv) < 4: 
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])

    if img is None :
        sys.exit("Could not read the images")

    if sys.argv[3] == '1':
        #negative
        img = 255 - img
    elif sys.argv[3] == '2':
        #intesity range
        img = img * (100/256) + 100
    elif sys.argv[3] == '3':
        #even line
        img[::2] = img[::2, ::-1]
    elif sys.argv[3] == '4':
        #line reflection
        dim = img.shape
        img[dim[0]//2:] = img[:dim[0]//2][::-1]
    elif sys.argv[3] == '5':
        #vertical reflection
        img = img[::-1] 

    cv.imwrite(sys.argv[2], img)

if __name__ == "__main__":
    main()