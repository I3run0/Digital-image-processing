import sys
import cv2 as cv
import matplotlib.pyplot as plt

def main(argv):
    # Check if the number of command-line arguments is correct
    if len(argv) < 2:
        print("Usage: python histogram.py <input_img_path> <output_hist_path>")
        return

    # Extract input and output file paths from command-line arguments
    input_img_path = argv[0]
    output_hist_path = argv[1]

    # Read the input image in grayscale
    img = cv.imread(input_img_path, cv.IMREAD_GRAYSCALE)

    # Compute the histogram of the input image
    plt.hist(img.ravel(), 256, [0, 256])

    # Save the histogram plot to the output file path
    plt.savefig(output_hist_path)

if __name__ == '__main__':
    # Call the main function with command-line arguments excluding the script name
    main(sys.argv[1:])

