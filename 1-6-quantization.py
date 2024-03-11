import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3: 
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])
    qtlevel = int(sys.argv[3])
    if img is None :
        sys.exit("Could not read the images")

    img = np.uint8(img/255 * qtlevel) * round(255/qtlevel)
    cv.imwrite(sys.argv[2], img) 

if __name__ == "__main__":
    main()