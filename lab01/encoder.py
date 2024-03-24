import sys
import cv2 as cv
import numpy as np

def main():
    # Check if all required parameters are provided
    if len(sys.argv) < 5:
        sys.exit("Usage: python encode.py input_image.png input_text.txt bit_plane output_image.png")

    # Extract command-line arguments
    input_image_path = sys.argv[1]
    input_text_path = sys.argv[2]
    max_bit_plane = int(sys.argv[3])
    output_image_path = sys.argv[4] 

    # Validate max_bit_plane
    if max_bit_plane not in (0, 1, 2):
        sys.exit("Invalid bit_plane value. Should be 0, 1, or 2.")

    # Read the input image
    img = cv.imread(input_image_path)
    if img is None:
        sys.exit("Could not read the image")

    # Read the text to be encoded
    with open(input_text_path, 'r') as file:
        text = file.read()

    # Convert text to binary ASCII representation
    text_ascii = [ord(char) for char in text] + [0]  # Add null terminator
    text_bin = np.unpackbits(np.array(text_ascii, dtype="uint8"))

    # Flatten the image into a 1D array
    img_encoded = np.reshape(img, -1).astype(np.uint8)
    # Embed text into the least significant bits of image pixels
    i = 0
    bit_plane = 0
    while (bit_plane + 1) * img_encoded.shape[0] < text_bin.shape[0] and not(bit_plane > max_bit_plane):
        img_encoded &= ~np.uint8(1 << bit_plane) 
        img_encoded |= (text_bin[i:i+img_encoded.shape[0]] << bit_plane)
        bit_plane += 1
        i += img_encoded.shape[0]

    # Fill the remaining text into the image
    if not(bit_plane > max_bit_plane):
        remaining_text_len = text_bin[i:].shape[0]
        img_encoded[:remaining_text_len] &= ~np.uint8((1 << bit_plane)) 
        img_encoded[:remaining_text_len] |= (text_bin[i:] << bit_plane)

    # Reshape the encoded image to its original shape
    img_encoded = np.reshape(img_encoded, img.shape).astype(np.uint8)

    # Write the encoded image to the output file
    cv.imwrite(output_image_path, img_encoded)

if __name__ == "__main__":
    main()
