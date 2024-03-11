import sys
import cv2 as cv
import numpy as np

def main():
    if len(sys.argv) < 3: 
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])
    exp = 1 / float(sys.argv[3])
    if img is None :
        sys.exit("Could not read the images")

    img = img * (1/255)
    img = np.power(img, exp)
    img = img * 255
    
    cv.imwrite(sys.argv[2], img)

if __name__ == "__main__":
    main()