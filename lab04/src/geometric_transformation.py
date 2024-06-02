import numpy as np
import cv2 as cv
from getopt import getopt, GetoptError
import sys

def scale_coordinates(new_width_indices, new_height_indices, scale_x, scale_y):
    """
    Scale coordinates (new_width_indices, new_height_indices) by scale_x and scale_y.
    """
    orig_x = new_width_indices / scale_x
    orig_y = new_height_indices / scale_y
    return orig_x, orig_y

def rotate_coordinates_vectorized(x, y, angle, cx, cy):
    """
    Rotate coordinates (x, y) around center (cx, cy) by angle (in radians).
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    x -= cx
    y -= cy

    x_new = cos_angle * x - sin_angle * y + cx
    y_new = sin_angle * x + cos_angle * y + cy

    return x_new, y_new

def lagrange_interpolation(image, x, y):
    original_height, original_width = image.shape[:2]
    x_int = np.floor(x).astype(int)
    y_int = np.floor(y).astype(int)
    x_frac = x - x_int
    y_frac = y - y_int
    x_int = np.clip(x_int, 1, original_width - 3)
    y_int = np.clip(y_int, 1, original_height - 3)

    def lagrange_weights(frac):
        L = np.zeros((frac.shape[0], frac.shape[1], 4))
        L[:, :, 0] = -frac * (frac - 1) * (frac - 2) / 6
        L[:, :, 1] = (frac + 1) * (frac - 1) * (frac - 2) / 2
        L[:, :, 2] = -frac * (frac + 1) * (frac - 2) / 2
        L[:, :, 3] = frac * (frac + 1) * (frac - 1) / 6
        return L

    def gather_pixel_values(image, y_int, x_int):
        pixel_values = np.zeros((y_int.shape[0], y_int.shape[1], 4, 4, 3))
        for i in range(4):
            for j in range(4):
                pixel_values[:, :, i, j] = image[
                    np.clip(y_int + i - 1, 0, image.shape[0] - 1),
                    np.clip(x_int + j - 1, 0, image.shape[1] - 1)
                ]
        return pixel_values

    L_x = lagrange_weights(x_frac)
    L_y = lagrange_weights(y_frac)

    pixel_values = gather_pixel_values(image, y_int, x_int)
    result = np.zeros((x.shape[0], x.shape[1], 3))

    for channel in range(3):
        result[..., channel] = np.sum(L_y[:, :, :, None] * pixel_values[..., channel] * L_x[:, :, None, :], axis=(2, 3))

    return np.clip(result, 0, 255)

def bicubic_interpolation(image, x, y):
    original_height, original_width = image.shape[:2]
    x_int = np.floor(x).astype(int)
    y_int = np.floor(y).astype(int)
    x_frac = x - x_int
    y_frac = y - y_int
    x_int = np.clip(x_int, 1, original_width - 3)
    y_int = np.clip(y_int, 1, original_height - 3)

    def P(a):
        return np.where(a > 0, a, 0)

    def R(s):
        return (1 / 6) * (P(s + 2) ** 3 - 4 * P(s + 1) ** 3 + 6 * P(s) ** 3 - 4 * P(s - 1) ** 3)

    result = np.zeros((x.shape[0], x.shape[1], 3))

    for channel in range(3):
        for m in range(-1, 3):
            for n in range(-1, 3):
                xm = np.clip(x_int + m, 0, original_width - 1)
                yn = np.clip(y_int + n, 0, original_height - 1)
                weight = R(m - x_frac) * R(n - y_frac)
                result[..., channel] += weight * image[yn, xm, channel]

    return np.clip(result, 0, 255)

def bilinear_interpolation(image, x, y):
    original_height, original_width = image.shape[:2]
    x1 = np.floor(x).astype(int)
    y1 = np.floor(y).astype(int)
    x2 = np.clip(x1 + 1, 0, original_width - 1)
    y2 = np.clip(y1 + 1, 0, original_height - 1)
    x_frac = x - x1
    y_frac = y - y1

    result = np.zeros((x.shape[0], x.shape[1], 3))

    for channel in range(3):
        result[..., channel] = (
            (1 - x_frac) * (1 - y_frac) * image[y1, x1, channel] +
            x_frac * (1 - y_frac) * image[y1, x2, channel] +
            (1 - x_frac) * y_frac * image[y2, x1, channel] +
            x_frac * y_frac * image[y2, x2, channel]
        )

    return np.clip(result, 0, 255)

def nearest_neighbor(image, x, y):
    x = np.clip(np.round(x).astype(int), 0, image.shape[1] - 1)
    y = np.clip(np.round(y).astype(int), 0, image.shape[0] - 1)
    return image[y, x]

def apply_scaling(image, scale_x, scale_y, interpolation_func):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_x)
    new_height = int(original_height * scale_y)

    # Create a grid of new coordinates
    new_height_indices, new_width_indices = np.indices((new_height, new_width))

    # Scale coordinates
    orig_x, orig_y = scale_coordinates(new_width_indices, new_height_indices, scale_x, scale_y)

    # Check if coordinates are within valid range
    valid_x = (orig_x >= 0) & (orig_x < original_width)
    valid_y = (orig_y >= 0) & (orig_y < original_height)
    valid_mask = valid_x & valid_y

    # Apply interpolation
    scaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    valid_pixels = interpolation_func(image, orig_x, orig_y).astype(np.uint8)
    print(valid_pixels)
    scaled_image[valid_mask] = valid_pixels[valid_mask]

    return scaled_image

def apply_rotation(image, angle, interpolation_func):
    original_height, original_width = image.shape[:2]
    cx, cy = original_width / 2, original_height / 2

    y, x = np.indices((original_height, original_width), dtype=np.float32)
    x, y = rotate_coordinates_vectorized(x, y, -angle, cx, cy)  # Negative angle for reverse mapping

    # Check if coordinates are within valid range
    valid_x = (x >= 0) & (x < original_width)
    valid_y = (y >= 0) & (y < original_height)
    valid_mask = valid_x & valid_y

    # Apply interpolation
    rotated_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    valid_pixels = interpolation_func(image, x, y).astype(np.uint8)

    rotated_image[valid_mask] = valid_pixels[valid_mask]

    return rotated_image

def rescale_and_rotate_image_lagrange(image, scale_x, scale_y, angle):
    scaled_image = apply_scaling(image, scale_x, scale_y, lagrange_interpolation)
    rotated_image = apply_rotation(scaled_image, angle, lagrange_interpolation)
    return rotated_image

def rescale_and_rotate_image_bilinear(image, scale_x, scale_y, angle):
    scaled_image = apply_scaling(image, scale_x, scale_y, bilinear_interpolation)
    rotated_image = apply_rotation(scaled_image, angle, bilinear_interpolation)
    return rotated_image

def rescale_and_rotate_image_nearest_neighbor(image, scale_x, scale_y, angle):
    scaled_image = apply_scaling(image, scale_x, scale_y, nearest_neighbor)
    rotated_image = apply_rotation(scaled_image, angle, nearest_neighbor)
    return rotated_image

def rescale_and_rotate_image_bicubic(image, scale_x, scale_y, angle):
    scaled_image = apply_scaling(image, scale_x, scale_y, bicubic_interpolation)
    rotated_image = apply_rotation(scaled_image, angle, bicubic_interpolation)
    return rotated_image

def main(argv: list):
    angle = 0  # Default angle
    scale_factor = 1.0  # Default scale factor
    interpolation_method = 'nearest'  # Default interpolation method

    try:
        opts, args = getopt(argv, "a:s:m:", ["angle=", "scale-factor=", "method="])
    except GetoptError as err:
        print(err)
        print("Usage: script.py <input_image_path> <output_image_path> [-a | --angle] [-s | --scale-factor] [-m | --method]")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-a", "--angle"):
            angle = float(arg)
        elif opt in ("-s", "--scale-factor"):
            scale_factor = float(arg)
        elif opt in ("-m", "--method"):
            interpolation_method = arg.lower()

    if len(args) < 2:
        print("Usage: script.py <input_image_path> <output_image_path> [-a | --angle] [-s | --scale-factor] [-m | --method]")
        sys.exit(2)
    
    input_image_path = args[0]
    output_image_path = args[1]

    # Read the input image
    image = cv.imread(input_image_path)
    if image is None:
        print(f"Error: Could not open or find the image '{input_image_path}'")
        sys.exit(2)

    # Select the interpolation method
    if interpolation_method == 'nearest':
        rescaled_image = rescale_and_rotate_image_nearest_neighbor(image, scale_factor, scale_factor, np.radians(angle))
    elif interpolation_method == 'bilinear':
        rescaled_image = rescale_and_rotate_image_bilinear(image, scale_factor, scale_factor, np.radians(angle))
    elif interpolation_method == 'bicubic':
        rescaled_image = rescale_and_rotate_image_bicubic(image, scale_factor, scale_factor, np.radians(angle))
    elif interpolation_method == 'lagrange':
        rescaled_image = rescale_and_rotate_image_lagrange(image, scale_factor, scale_factor, np.radians(angle))
    else:
        print(f"Error: Unknown interpolation method '{interpolation_method}'")
        print("Available methods: nearest, bilinear, bicubic, lagrange")
        sys.exit(2)

    # Save the output image
    cv.imwrite(output_image_path, rescaled_image)
    print(f"Output image saved as '{output_image_path}'")

if __name__ == "__main__":
    main(sys.argv[1:])
