import sys
import cv2 as cv
import numpy as np

BITS_NUMBER = 7

def main():
    # Check if all required parameters are provided
    if len(sys.argv) < 4:
        sys.exit("Usage: python decode.py input_image.png max_bit_plane output_text.txt")

    # Extract command-line arguments
    input_image_path = sys.argv[1]
    max_bit_plane = int(sys.argv[2])
    output_text_path = sys.argv[3]

    # Read the input image
    img = cv.imread(input_image_path)
    if img is None:
        sys.exit("Could not read the image")

    # Extract the least significant bit from each pixel's
    img_bin = np.reshape(np.unpackbits(img, axis=2), (img.shape[0] * img.shape[1] * img.shape[2], 8))
    text_bin = np.transpose(img_bin[:, BITS_NUMBER - max_bit_plane:][:, ::-1])
    text_bin = np.reshape(text_bin, text_bin.shape[0] * text_bin.shape[1])

    # Convert binary text to ASCII characters
    text_ascii = np.packbits(text_bin)
    text = ''.join(chr(lttr) for lttr in text_ascii)

    # Find the null terminator and extract text before it
    null_index = text.find('\0')

    if null_index != -1:
        text = text[:null_index]

    # Write the decoded text to the output file
    with open(output_text_path, 'w') as file:
        file.write(text)

if __name__ == "__main__":
    main()
