import sys
import cv2 as cv
import numpy as np


def main():
    if len(sys.argv) < 4:
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])
    
    if img is None :
        sys.exit("Could not read the images")

    bit = int(sys.argv[3])
    print(bit)
    img = ((img >> bit) & 1) * 255
    cv.imwrite(sys.argv[2], img)


if __name__ == "__main__":
    main()