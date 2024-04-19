import sys
import os
import cv2 as cv
import numpy as np
import fft_utils as uttf
from getopt import getopt

def get_circle_mask(radius, center_coords, shape, fill):
    """
    Generate a circular mask.

    Args:
        radius (int): Radius of the circle.
        center_coords (tuple): Coordinates of the circle's center.
        shape (tuple): Shape of the image.
        fill (bool): Whether to fill the circle or not.

    Returns:
        numpy.ndarray: Binary mask representing the circle.
    """
    mask = np.zeros(shape, dtype='uint8') if fill else np.ones(shape, dtype='uint8')
    mask = cv.circle(mask, center_coords, radius, 1 if fill else 0, -1)
    return mask

def apply_lower_pass_filter(fft, radius):
    """
    Apply a lower pass filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        radius (int): Radius for the filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    mask = get_circle_mask(radius, (fft.shape[0] // 2, fft.shape[1] // 2), fft.shape, True)
    filtered_fft = fft * mask
    return filtered_fft

def apply_higher_pass_filter(fft, radius):
    """
    Apply a higher pass filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        radius (int): Radius for the filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    mask = get_circle_mask(radius, (fft.shape[0] // 2, fft.shape[1] // 2), fft.shape, False)
    filtered_fft = fft * mask
    return filtered_fft

def apply_band_pass_filter(fft, radius_higher, radius_lower):
    """
    Apply a band pass filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        radius_higher (int): Radius for the higher pass filter.
        radius_lower (int): Radius for the lower pass filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    filtered_fft = apply_higher_pass_filter(fft, radius_higher)
    filtered_fft = apply_lower_pass_filter(filtered_fft, radius_lower)
    return filtered_fft

def apply_reject_band_filter(fft, radius_higher, radius_lower):
    """
    Apply a reject band filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        radius_higher (int): Radius for the higher pass filter.
        radius_lower (int): Radius for the lower pass filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    filtered_fft = apply_higher_pass_filter(fft, radius_higher)
    mask = get_circle_mask(radius_lower, (fft.shape[0] // 2, fft.shape[1] // 2), fft.shape, True)
    filtered_fft[mask == 1] = fft[mask == 1]
    return filtered_fft

def main(argv):
    # Parse command line options
    opts, args = getopt(argv, "m:p:s", ["method=", "parameters=", "spec"])
    method = None
    parameters = None
    img_path = None
    gen_spec = False
    out_dir = "output_images"
    img_ext = "png"

    # Process command line options
    for opt, arg in opts:
        if opt in ("-m", "--method"):
            method = arg
        elif opt in ("-p", "--parameters"):
            parameters = arg.split(',')
        elif opt in ("-s", "--spec"):
            gen_spec = True
    
    # Check if method and image path are provided
    if method is None or len(args) < 1:
        print("Method and image path are required.")
        print("Usage: python script.py -m <method> -p <parameters> <image_path>")
        sys.exit(2)

    # Extract image path
    img_path = args[0]
    img_name = os.path.basename(img_path).split(".")[0]
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Apply FFT to the image
    fft = uttf.get_fft_from_img(img)
    mag = uttf.get_mag_from_fft(fft)

    # Choose filtering method and apply filter
    if method == 'low':
        filtered_fft = apply_lower_pass_filter(fft, int(parameters[0]))
        filtered_mag = apply_lower_pass_filter(mag, int(parameters[0]))
    elif method == 'high':
        filtered_fft = apply_higher_pass_filter(fft, int(parameters[0]))
        filtered_mag = apply_higher_pass_filter(mag, int(parameters[0]))
    elif method == 'band':
        filtered_fft = apply_band_pass_filter(fft, int(parameters[0]), int(parameters[1]))
        filtered_mag = apply_band_pass_filter(mag, int(parameters[0]), int(parameters[1]))
    elif method == 'reject':
        filtered_fft = apply_reject_band_filter(fft, int(parameters[0]), int(parameters[1]))
        filtered_mag = apply_reject_band_filter(mag, int(parameters[0]), int(parameters[1]))
    
    # Reconstruct image from filtered FFT
    img_back = uttf.get_img_from_fft(filtered_fft)
   
    # Create output directory if it doesn't exist
    if not(os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    
    # Create method-specific subdirectory
    if not(os.path.isdir(f'{out_dir}/{method}')):
        os.mkdir(f'{out_dir}/{method}')

    # Save filtered image
    out_path = f'{out_dir}/{method}/{img_name}_{method}_{"_".join(parameters)}'
    cv.imwrite(f'{out_path}.{img_ext}', img_back)

    # Optionally, generate and save magnitude spectrum images
    if gen_spec:
        cv.imwrite(f'{out_path}_spec_cropped.{img_ext}', filtered_mag)
        
    
if __name__ == "__main__":
    main(sys.argv[1:])
