import sys
import cv2 as cv
import fft_utils as uts

def main(argv):
    # Check if there are enough command-line arguments
    if len(argv) < 2:
        print("Usage: python script.py <input_image_path> <output_image_path>")
        sys.exit(1)

    # Extract command-line arguments
    input_image_path = argv[0]
    output_image_path = argv[1]

    # Read the input image in grayscale
    input_image = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
    
    # Apply FFT to the input image
    fft = uts.get_fft_from_img(input_image)
    
    # Get the magnitude from the FFT
    f_mag = uts.get_mag_from_fft(fft)

    # Write the processed image to the output file
    output_filename = output_image_path.split('.')[0] + '_mag.png'
    cv.imwrite(output_filename, f_mag)

if __name__ == '__main__':
    main(sys.argv[1:])
