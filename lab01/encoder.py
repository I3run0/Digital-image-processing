import sys
import cv2 as cv
import numpy as np

def main():

    if len(sys.argv) < 4:
        sys.exit("Missing paramenters")

    #instance the image
    img = cv.imread(sys.argv[1])
    if img is None:
        sys.exit("Could not read the image")

    #Get the text binary version
    with open(sys.argv[2], 'r') as file:
        text = file.read()

    text += "\0"
    text_ascii = [ord(lttr) for lttr in text]
    text_bin_raw = np.unpackbits(np.array(text_ascii, dtype="uint8"))
    text_bin_raw_sz = np.shape(text_bin_raw)[0]
    #Enconding text to the image
    x, y, z = img.shape
    img_bin_raw_mask = np.zeros((x * z * y * 8), dtype='uint8')
    img_bin_raw_mask[:text_bin_raw_sz] = 1
    img_bin_raw_mask = np.reshape(img_bin_raw_mask, (8, x * y * z))

    img_bin = np.reshape(np.unpackbits(img, axis=2), (x * z * y, 8))
    text_bin = np.resize(text_bin_raw, (8, (x * z * y)))

    img_bin[:, 7] = text_bin[0][img_bin_raw_mask[0]]

    img = np.reshape(np.packbits(img_bin, axis=1), (x, y, z))

    cv.imwrite(sys.argv[4], img)

def main2():
    if len(sys.argv) < 4:
        sys.exit("Missing paramenters")

    #instance the image
    img = cv.imread(sys.argv[1])
    if img is None:
        sys.exit("Could not read the image")

    #Get the text binary version
    with open(sys.argv[2], 'r') as file:
        text = file.read()

    text += "\0"
    text_ascii = [ord(lttr) for lttr in text]
    text_bin = np.unpackbits(np.array(text_ascii, dtype="uint8"))
    
    text_size = np.shape(text_bin)[0]
    img_pxs_sz = img.shape[0] * img.shape[1]
    print(np.shape(text_bin)[0], 512 **2)
    #r = np.where(img[:, :, 0] & 1 == text_bin, img[:, :, 0], text_bin)
    #print(r)

if __name__ == "__main__":
    main()