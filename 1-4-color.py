import sys
import cv2 as cv
import numpy as np

def main():
    if len(sys.argv) < 4: 
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])

    if img is None :
        sys.exit("Could not read the images")

    img = img.astype(np.uint16)

    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    if sys.argv[3] == "1":
        img[:,:,0] = R * 0.393 + G * 0.769 + B * 0.189
        img[:,:,1] = R * 0.349 + G * 0.686 + B * 0.168
        img[:,:,2] = R * 0.272 + G * 0.534 + B * 0.131

    elif sys.argv[3] == "2":
        I = R * 0.2989 + G * 0.5870 + B * 0.1140
        img[:,:,0], img[:,:,1], img[:,:,2] = I, I, I
        
    img = img.clip(max=255).astype(np.uint8)

    cv.imwrite(sys.argv[2], img)

if __name__ == "__main__":
    main()