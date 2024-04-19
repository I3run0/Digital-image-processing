import sys
import os
import numpy as np
import cv2 as cv
import fft_utils as uts


def main(argv):
    # Check if there are enough command-line arguments
    if len(argv) < 3:
        print("Usage: python script.py <compression_level> <input_image_path> <output_image_path>")
        sys.exit(1)

    # Extract command-line arguments
    compression_level = float(argv[0])
    input_image_path = argv[1]
    output_image_path = argv[2]

    # Ensure compression level is valid
    if compression_level > 1:
        print("Compression level should be less than or equal to 1.")
        sys.exit(1)

    # Read the input image in grayscale
    input_image = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)

    # Apply FFT to the input image
    fft = uts.get_fft_from_img(input_image)

    # Determine the threshold value based on compression level
    threshold_index = round((1 - compression_level) * (fft.size - 1))
    sorted_fft = np.sort(np.abs(fft).flatten())[::-1]
    threshold = sorted_fft[threshold_index]

    # Thresholding: set FFT coefficients below the threshold to zero
    compressed_fft = np.where(np.abs(fft) >= threshold, fft, 0)

    # Reconstruct the image from the modified FFT
    output_image = uts.get_img_from_fft(compressed_fft)
    
    # Write the processed image to the output file
    cv.imwrite(output_image_path, output_image)

if __name__ == '__main__':
    main(sys.argv[1:])
