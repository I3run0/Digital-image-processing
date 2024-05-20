import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import alignmnet_utils as ut
import sys
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
    angles = np.fromiter([objective_function(img, i) for i in range(0, 180, 1)], dtype=np.int64)
    best_index = np.where(angles == np.max(angles))[0][0]
    print(np.where(angles == np.max(angles)))
    best_angle = best_index if best_index < 90 else best_index - 180 
    return best_angle

def save_histogram(image, filename):
    """
    Save the histogram of the input image as a PDF file.
    
    Args:
        image (numpy.ndarray): Input image.
        filename (str): File name for saving the histogram.
    """
    hist, bins = np.histogram(image.ravel(), bins=256, range=[0,256])
    plt.plot(hist, color='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
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
    input_image_path = None
    output_image_path = None
    plot_flag = False
    histogram_flag = False

    opts, args = getopt(argv, "ph", ["plot", "histogram"])
    for opt, _ in opts:
        if opt in ("-p", "--plot"):
            plot_flag = True
        elif opt in ("-h", "--histogram"):
            histogram_flag = True

    input_image_path = args[0]
    output_image_path = args[1]

    # Read input image
    img = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
    
    # Find the best rotation angle
    best_angle = find_best_rotation_angle(img)
    
    # Rotate the image using the best angle
    rotated_image = ut.rotate_image(img, best_angle)
    
    # If histogram flag is set, save histograms
    if histogram_flag:
        save_histogram(img, "original_histogram.pdf")
        save_histogram(rotated_image, "rotated_histogram.pdf")
    
    # If plot flag is set, display original and rotated images
    if plot_flag:
        plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(rotated_image, cmap='gray')
        plt.title(f'Rotated Image ({best_angle} Degrees)'), plt.xticks([]), plt.yticks([])
        plt.show()

    # Save the rotated image
    cv.imwrite(output_image_path, rotated_image)

    # Print output message
    print_output_message(input_image_path, output_image_path, best_angle)

if __name__ == "__main__":
    main(sys.argv[1:])
