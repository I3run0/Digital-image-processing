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
    tk_bin_sz = tk_bin.shape[0]
    print(tk_bin_sz)
    x, y, z = img.shape
    img_bin = np.reshape(np.unpackbits(img, axis=2), (x * z * y , 8))
    print(img_bin.shape)
    lower_bit = img_bin[:, 7]
    print(lower_bit.shape)
    for i in range(lower_bit.shape[0] - tk_bin_sz):
        if (lower_bit[i:i + tk_bin_sz] == tk_bin).all():
            break
        
    

        


if __name__ == "__main__":
    main()