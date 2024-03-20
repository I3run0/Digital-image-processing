import sys
import cv2 as cv
import numpy as np

def main():

    if len(sys.argv) < 3:
        sys.exit("Missing paramenters")

    #instance the image
    img = cv.imread(sys.argv[1])
    if img is None:
        sys.exit("Could not read the image")
    
    #decoding
    tk = "\fim"
    tk_bin = np.unpackbits(np.array([ord(lttr) for lttr in tk], dtype='uint8'))
    print(tk_bin)
    x, y, z = img.shape
    img_bin = np.reshape(np.unpackbits(img, axis=2), (x * z * y, 8))

    print(img_bin[:, 7])
    print(np.where(img_bin == tk_bin, True, False))


if __name__ == "__main__":
    main()