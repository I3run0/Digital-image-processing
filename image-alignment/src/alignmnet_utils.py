
import cv2 as cv
import numpy as np
import pytesseract as ocr
import difflib as diff
import os

def rotate_image(img, angle):
    """
    Rotate the image by a given angle without clipping the content.
    
    Args:
        img (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        
    Returns:
        numpy.ndarray: Rotated image.
    """
    height, width = img.shape[:2]
    new_height = int(width * np.abs(np.sin(np.radians(angle))) + height * np.abs(np.cos(np.radians(angle))))
    new_width = int(height * np.abs(np.sin(np.radians(angle))) + width * np.abs(np.cos(np.radians(angle))))
    rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    rotated_image = cv.warpAffine(img, rotation_matrix, (new_width, new_height), flags=cv.INTER_LINEAR)
    return rotated_image

def compare_ocr_tesseract(input_image_path, output_image_path, path_to_save = None):
    """
    Compare OCR results using Tesseract on the input and output images.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to the output image.
    """
    # Perform OCR using Tesseract on the input image
    input_ocr = ocr.image_to_string(cv.imread(input_image_path))

    # Perform OCR using Tesseract on the output image
    output_ocr = ocr.image_to_string(cv.imread(output_image_path))


    # Compute the difference between the OCR texts
    df = diff.ndiff(input_ocr.splitlines(keepends=True), output_ocr.splitlines(keepends=True))
    diff_text = ''.join(df)

    input_ocr = '> ' + input_ocr.replace('\n', '\n> ')
    output_ocr = '> ' + output_ocr.replace('\n', '\n> ')
        
    print("********************* OCR Comparison ************************")
    print("Input OCR:")
    print(input_ocr)
    print("Output OCR:")
    print(output_ocr)
    print("Difference:")
    print(diff_text)

    if path_to_save:
        path_to_save = f'{os.getcwd()}/{path_to_save}'
        with open(path_to_save, "w") as file:
            file.write("Input OCR:\n")
            file.write(input_ocr + "\n")
            file.write("Output OCR:\n")
            file.write(output_ocr + "\n")
            file.write("Difference:\n")
            file.write(diff_text + "\n")
        
        file.close()

def print_output_message(input_image_path, output_image_path, best_angle, heuristic):
    """
    Print an output message explaining where the file was saved,
    the input image path, the best rotation angle, and the heuristic used to obtain it.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to the saved output image.
        best_angle (int): Best rotation angle.
        heuristic (str): Heuristic used for rotation.
    """
    print("******************** Execution Information *********************")
    print(f"Input image: '{input_image_path}'")
    print(f"Output image saved as '{output_image_path}'")
    print(f"Best rotation angle: {best_angle} degrees")
    print(f"Heuristic used: {heuristic}")