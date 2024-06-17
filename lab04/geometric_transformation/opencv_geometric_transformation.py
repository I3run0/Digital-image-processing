import argparse
import cv2 as cv
import numpy as np
import sys

# Function to apply scaling using OpenCV
def apply_scaling_opencv(image: np.ndarray, scale_x: float, scale_y: float, interpolation_method):
    """
    Scales the input image using the specified scaling factors and interpolation method.

    Parameters:
    - image: Input image as a NumPy array.
    - scale_x: Scaling factor for the width.
    - scale_y: Scaling factor for the height.
    - interpolation_method: Interpolation method to be used (e.g., cv.INTER_NEAREST, cv.INTER_LINEAR, etc.).

    Returns:
    - scaled_image: Scaled image as a NumPy array.
    """
    height, width = image.shape[:2]
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    scaled_image = cv.resize(image, (new_width, new_height), interpolation=interpolation_method)
    return scaled_image

# Function to apply rotation using OpenCV
def apply_rotation_opencv(image: np.ndarray, angle: float, interpolation_method: int):
    """
    Rotates the input image using the specified angle and interpolation method.

    Parameters:
    - image: Input image as a NumPy array.
    - angle: Angle to rotate the image (in radians).
    - interpolation_method: Interpolation method to be used (e.g., cv.INTER_NEAREST, cv.INTER_LINEAR, etc.).

    Returns:
    - rotated_image: Rotated image as a NumPy array.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)  # Calculate the center of the image for rotation
    rotation_matrix = cv.getRotationMatrix2D(center, np.degrees(angle), 1.0)  # Get the rotation matrix
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height), flags=interpolation_method)  # Rotate the image
    return rotated_image

# Main function logic for choosing OpenCV
INTERPOLATION_METHODS_OPENCV = {
    'nearest': cv.INTER_NEAREST,
    'bilinear': cv.INTER_LINEAR,
    'bicubic': cv.INTER_CUBIC,
    'lanczos': cv.INTER_LANCZOS4
}

def main() -> None:
    """
    Main function to handle command-line arguments and perform image scaling or rotation.

    Parameters:
    - argv: List of command-line arguments
    """
    parser = argparse.ArgumentParser(description="Image scaling and rotation script")
    subparsers = parser.add_subparsers(dest="command")

    # Rotate command
    parser_rotate = subparsers.add_parser("rotate", help="Rotate an image")
    parser_rotate.add_argument("input_image_path", type=str, help="Input image path")
    parser_rotate.add_argument("output_image_path", type=str, help="Output image path")
    parser_rotate.add_argument("-a", "--angle", type=float, default=0, help="Angle to rotate (in degrees)")
    parser_rotate.add_argument("-m", "--method", type=str, choices=INTERPOLATION_METHODS_OPENCV.keys(), default='nearest', help="Interpolation method")

    # Scale command
    parser_scale = subparsers.add_parser("scale", help="Scale an image")
    parser_scale.add_argument("input_image_path", type=str, help="Input image path")
    parser_scale.add_argument("output_image_path", type=str, help="Output image path")
    parser_scale.add_argument("-s", "--scale-factor", type=float, default=1.0, help="Scaling factor")
    parser_scale.add_argument("-wh", "--width-height", type=int, nargs=2, default=None, help="Width and height")
    parser_scale.add_argument("-m", "--method", type=str, choices=INTERPOLATION_METHODS_OPENCV.keys(), default='nearest', help="Interpolation method")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Read the input image
    input_image: np.ndarray = cv.imread(args.input_image_path)
    if input_image is None:
        print(f"Error: Could not open or find the image '{args.input_image_path}'")
        sys.exit(2)

    output_image: np.ndarray = None
    if args.command == "rotate":
        angle = np.radians(args.angle)
        output_image = apply_rotation_opencv(image=input_image, angle=angle, interpolation=INTERPOLATION_METHODS_OPENCV[args.method])
    elif args.command == "scale":
        scale_x, scale_y =  (args.width_height[0]/input_image.shape[0], args.width_height[1]/input_image.shape[1]) if args.width_height\
              else (args.scale_factor, args.scale_factor)
        output_image = apply_scaling_opencv(image=input_image, scale_x=scale_x, scale_y=scale_y, interpolation=INTERPOLATION_METHODS_OPENCV[args.method])
    else:
        parser.print_help()
        sys.exit(2)

    # Save the output image
    cv.imwrite(args.output_image_path, output_image)
    print(f"Output image saved as '{args.output_image_path}'")

if __name__ == "__main__":
    main()
