import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import alignmnet_utils as ut
import sys
import os
from getopt import getopt

def objective_function(img, i):
    """
    Objective function to compute the score for a given rotation angle.
    
    Args:
        img (numpy.ndarray): Input image.
        i (int): Rotation angle.
        
    Returns:
        int: Score for the given rotation angle.
    """
    rotated_img = ut.rotate_image(img, i)
    rows_sum = np.sum(rotated_img, axis=1)
    return np.sum(np.diff(rows_sum) ** 2)

def find_best_rotation_angle(img):
    """
    Find the best rotation angle for the input image.
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        int: Best rotation angle.
    """
    angles = np.fromiter((objective_function(img, i) for i in range(0, 180, 1)), dtype=np.int64)
    best_index = np.argmax(angles)
    best_angle = best_index if best_index < 90 else best_index - 180
    return best_angle

def make_line_histogram(image, filename):
    """
    Save the line sum histogram of the image as a PDF file.
    
    Args:
        image (numpy.ndarray): Input image.
        filename (str): File name for saving the histogram.
    """
    rows_sum = np.sum(image, axis=1)
    
    plt.bar(np.arange(len(rows_sum)), rows_sum)
    plt.xlabel('Image lines')
    plt.ylabel('Pixel sum')
    plt.title('Line Sum Histogram')
    plt.savefig(filename, format='pdf')
    plt.close()

def print_output_message(input_image_path, output_image_path, best_angle):
    """
    Print an output message explaining where the file was saved,
    the input image path, the best rotation angle, and the heuristic used to obtain it.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to the saved output image.
        best_angle (int): Best rotation angle.
    """
    print()
    print(f"Input image: '{input_image_path}'")
    print(f"Output image saved as '{output_image_path}'.")
    print(f"Best rotation angle: {best_angle} degrees")
    print("Heuristic used: Maximization of the difference of squared row sums.")

def main(argv):
    """
    Main function to process input arguments and perform image rotation.
    """
    plot_flag = False
    histogram_flag = False

    opts, args = getopt(argv, "ph", ["plot", "histogram"])
    for opt, _ in opts:
        if opt in ("-p", "--plot"):
            plot_flag = True
        elif opt in ("-h", "--histogram"):
            histogram_flag = True

    if len(args) < 2:
        print("Usage: script.py <input_image_path> <output_image_path> [-p | --plot] [-h | --histogram]")
        sys.exit(2)

    input_image_path = args[0]
    input_file_name = os.path.basename(input_image_path).split(".")[0]
    output_image_path = args[1]
    output_file_name = os.path.basename(output_image_path).split(".")[0]
    output_dir = os.path.dirname(output_image_path)

    # Read input image
    img = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, f"Error: Could not read image '{input_image_path}'"
    r, bitwise_img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


    # Find the best rotation angle
    best_angle = find_best_rotation_angle(bitwise_img)
    
    # Rotate the image using the best angle
    rotated_image = ut.rotate_image(img, best_angle)
    bitwise_rotated = ut.rotate_image(bitwise_img, best_angle)
    
    # If plot flag is set, display original and rotated images
    if plot_flag:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(rotated_image, cmap='gray')
        plt.title(f'Rotated Image ({best_angle} Degrees)')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # If histogram flag is set, save histograms
    if histogram_flag:
        make_line_histogram(bitwise_rotated, f'{output_dir}/{output_file_name}_hist.pdf')
        make_line_histogram(bitwise_img, f'{output_dir}/{input_file_name}_hist.pdf')
    # Save the rotated image
    cv.imwrite(output_image_path, rotated_image)

    # Print output message
    print_output_message(input_image_path, output_image_path, best_angle)

if __name__ == "__main__":
    main(sys.argv[1:])
