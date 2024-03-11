import sys
import cv2 as cv
import numpy as np

def main():
    if len(sys.argv) < 6: 
        sys.exit("Parameters are missing")

    img1, img2 = cv.imread(sys.argv[1]), cv.imread(sys.argv[2])
    A, B = float(sys.argv[3]), float(sys.argv[4])
    
    if img1 is None and img2 is None:
        sys.exit("Could not read the images")

    newImage = (img1 * A + img2 * B) / (A + B)
    cv.imwrite(sys.argv[5], newImage)

if __name__ == "__main__":
    main()