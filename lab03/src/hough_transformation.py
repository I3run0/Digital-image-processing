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
        rho, theta = line
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

def objective_function(image):
    """
    Find main lines in the input image using Hough transform.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Array containing detected lines.
    """
    edges = cv.Canny(image, 100, 100, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 0)
    return lines[np.argmax(lines[:1])]
    
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
    image = cv.imread(input_image_path)
    assert image is not None, f"Error: Could not read image '{input_image_path}'"
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find main line using Hough transform
    detected_line = objective_function(gray_image)
    
    # Calculate the best rotation angle based on detected line
    best_rotation_angle = np.rad2deg(detected_line[0][1]) - 90

    # Draw detected line on the input image
    image_with_lines = draw_detected_lines(image, detected_line) if draw_lines_flag else None
    
    # Rotate the image using the calculated angle
    rotated_image = ut.rotate_image(image, best_rotation_angle)

    # Write images to files
    cv.imwrite(output_image_path, rotated_image)
    if draw_lines_flag:
        cv.imwrite(f'{output_dir}/{input_file_name}_with_lines.png', image_with_lines)

    # If plot flag is set, display original image, image with detected line, and rotated image
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

    ut.print_output_message(input_image_path, output_image_path, 
                         best_rotation_angle, "Mean angle of detected lines through hough transformation.")

    ut.compare_ocr_tesseract(input_image_path, output_image_path, f'{output_dir}/{input_file_name}_ocr.txt')
if __name__ == '__main__':
    main(sys.argv[1:])