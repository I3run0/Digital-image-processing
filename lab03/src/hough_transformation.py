import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import alignmnet_utils as ut
import sys
import os
from getopt import getopt

def draw_detected_lines(image, lines):
    """
    Draw detected lines on the input image.
    
    Args:
        image (numpy.ndarray): Input image.
        lines (numpy.ndarray): Array containing lines detected by Hough transform.
        
    Returns:
        numpy.ndarray: Image with detected lines drawn.
    """
    image_with_lines = np.copy(image)
    for line in lines:
        rho, theta = line[0]
        # Convert polar coordinates to Cartesian coordinates
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho
        x1 = int(x0 + 1000 * (-sin_theta))
        y1 = int(y0 + 1000 * (cos_theta))
        x2 = int(x0 - 1000 * (-sin_theta))
        y2 = int(y0 - 1000 * (cos_theta))
        # Draw the line on the image
        cv.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image_with_lines

def calculate_mean_angle(lines):
    """
    Calculate the mean angle of detected lines.
    
    Args:
        lines (numpy.ndarray): Array containing lines detected by Hough transform.
        
    Returns:
        float: Mean angle in degrees.
    """
    total_angle = sum(np.rad2deg(line[0][1]) for line in lines)
    return total_angle / len(lines)

def find_main_lines(image):
    """
    Find main lines in the input image using Hough transform.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Array containing detected lines.
    """
    edges = cv.Canny(image, 100, 100, apertureSize=3)
    k = 0
    lines = None
    while True:
        k += 1
        detected_lines = cv.HoughLines(edges, 1, np.pi / 180, k)
        if detected_lines is None:
            break
        lines = detected_lines
    return lines

def print_output_message(input_image_path, output_image_path, best_angle):
    """
    Print an output message explaining where the file was saved,
    the input image path, the best rotation angle, and the heuristic used to obtain it.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to the saved output image.
        best_angle (int): Best rotation angle.
    """
    print(f"Input image: '{input_image_path}'")
    print(f"Output image saved as '{output_image_path}'.")
    print(f"Best rotation angle: {best_angle} degrees")
    print("Heuristic used: Mean angle of detected lines through hough transformation.")

def main(argv):
    """
    Main function to process input arguments and perform image processing tasks.
    """
    input_image_path = None
    output_image_path = None
    plot_flag = False
    draw_lines_flag = False

    opts, args = getopt(argv, "pd", ["plot", "draw-lines"])
    for opt, _ in opts:
        if opt in ("-p", "--plot"):
            plot_flag = True
        elif opt in ("-d", "--draw-lines"):
            draw_lines_flag = True

    input_image_path = args[0]
    input_file_name = os.path.basename(input_image_path).split(".")[0]
    output_image_path = args[1]
    output_dir = os.path.dirname(output_image_path)

    # Read input image
    image = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
    assert image is not None, f"Error: Could not read image '{input_image_path}'"

    # Find main lines using Hough transform
    detected_lines = find_main_lines(image)
    
    # Calculate the best rotation angle based on detected lines
    best_rotation_angle = calculate_mean_angle(detected_lines) - 90

    # Draw detected lines on the input image
    image_with_lines = draw_detected_lines(image, detected_lines) if draw_lines_flag else None
    
    # Rotate the image using the calculated angle
    rotated_image = ut.rotate_image(image, best_rotation_angle)

    # Write images to files
    cv.imwrite(output_image_path, rotated_image)
    if draw_lines_flag:
        cv.imwrite(f'{output_dir}/{input_file_name}_with_lines.png', image_with_lines)

    # If plot flag is set, display original image, image with detected lines, and rotated image
    if plot_flag:
        to_plot = [
            ['Original Image', image],
            [f'Rotated Image {round(best_rotation_angle)}', rotated_image]
        ]

        if draw_lines_flag:
            to_plot.append(['Detected Lines', image_with_lines])

        for i in range(1, len(to_plot) + 1):
            plt.subplot(1, len(to_plot), i), plt.imshow(to_plot[i - 1][1], cmap='gray')
            plt.title(to_plot[i - 1][0]), plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])