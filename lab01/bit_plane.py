import sys
import cv2 as cv
import numpy as np

def main():
    # Check if all required parameters are provided
    if len(sys.argv) < 4:
        sys.exit("Usage: python bit_plane.py input_image.png output_path (1 2 3 ...)")

    # Extract command-line arguments
    input_image = sys.argv[1]
    output_path = sys.argv[2]
    bit_plane = [int(i) for i in sys.argv[3:]]

    # Read the input image
    img = cv.imread(input_image)
    if img is None:
        sys.exit("Could not read the image")

    chanel_name = {
        0: 'R', 
        1: 'G', 
        2: 'B',
    }

    # Apply bit-plane slicing
    for i in range(3):
        for j in bit_plane:
            # Write the modified image to the output file
            img_bit_plane = np.zeros_like(img)
            img_bit_plane[:,:,i] = ((img[:,:,i] >> j) & 1) * 255
            cv.imwrite(f'{output_path}/{input_image.split(".")[0]}_{chanel_name[i]}_plane_{j}.png', img_bit_plane)

if __name__ == '__main__':
    main()
