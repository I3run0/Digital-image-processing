import sys, os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import fft_utils as uttf
from getopt import getopt

def get_circle_mask(r, cords, shape, fill):
    """
    Generate a circular mask.

    Args:
        r (int): Radius of the circle.
        cords (tuple): Coordinates of the circle's center.
        shape (tuple): Shape of the image.
        fill (bool): Whether to fill the circle or not.

    Returns:
        numpy.ndarray: Binary mask representing the circle.
    """
    mask = np.zeros(shape, dtype='uint8') if fill else np.ones(shape, dtype='uint8')
    mask = cv.circle(mask, cords, r, 1 if fill else 0, -1)
    return mask

def lower_pass_filter(fft, r):
    """
    Apply a lower pass filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        r (int): Radius for the filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    mask = get_circle_mask(r, (fft.shape[0] // 2, fft.shape[1] // 2), fft.shape, True)
    f_fft = fft * mask
    return f_fft

def higher_pass_filter(fft, r):
    """
    Apply a higher pass filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        r (int): Radius for the filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    mask = get_circle_mask(r, (fft.shape[0] // 2, fft.shape[1] // 2), fft.shape, False)
    f_fft = fft * mask
    return f_fft

def band_pass_filter(fft, r_higher, r_lower):
    """
    Apply a band pass filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        r_higher (int): Radius for the higher pass filter.
        r_lower (int): Radius for the lower pass filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    f_fft = higher_pass_filter(fft, r_higher)
    f_fft = lower_pass_filter(f_fft, r_lower)
    return f_fft

def reject_band_filter(fft, r_higher, r_lower):
    """
    Apply a reject band filter on the Fourier Transform.

    Args:
        fft (numpy.ndarray): Input Fourier Transform.
        r_higher (int): Radius for the higher pass filter.
        r_lower (int): Radius for the lower pass filter.

    Returns:
        numpy.ndarray: Filtered Fourier Transform.
    """
    f_fft = higher_pass_filter(fft, r_higher)
    mask = get_circle_mask(r_lower, (fft.shape[0] // 2, fft.shape[1] // 2), fft.shape, True)
    f_fft[mask == 1] = fft[mask == 1]
    return f_fft

def main(argv):
    opts, args = getopt(argv, "m:p:s", ["method=", "parameters=", "spec"])
    method = None
    parameters = None
    img_path = None
    gen_spec = False
    out_dir = "output_images"
    img_ext = "png"

    for opt, arg in opts:
        if opt in ("-m", "--method"):
            method = arg
        elif opt in ("-p", "--parameters"):
            parameters = arg.split(',')
        elif opt in ("-s", "--spec"):
            gen_spec = True
    
    if method is None or len(args) < 1:
        print("Method and image path are required.")
        print("Usage: python script.py -m <method> -p <parameters> <image_path>")
        sys.exit(2)

    img_path = args[0]
    img_name = os.path.basename(img_path).split(".")[0]
    if not(os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    f_fft, f_mag = None, None
    
    fft = uttf.get_fft_from_img(img)
    mag = uttf.get_mag_from_fft(fft)
    if method == 'low':
        f_fft = lower_pass_filter(fft, int(parameters[0]))
        f_mag = lower_pass_filter(mag, int(parameters[0]))
    elif method == 'high':
        f_fft = higher_pass_filter(fft, int(parameters[0]))
        f_mag = higher_pass_filter(mag, int(parameters[0]))
    elif method == 'band':
        f_fft = band_pass_filter(fft, int(parameters[0]), int(parameters[1]))
        f_mag = band_pass_filter(mag, int(parameters[0]), int(parameters[1]))
    elif method == 'reject':
        f_fft = reject_band_filter(fft, int(parameters[0]), int(parameters[1]))
        f_mag = reject_band_filter(mag, int(parameters[0]), int(parameters[1]))
    
    img_back = uttf.get_img_from_fft(f_fft)

    '''
    plt.subplot(121)s
    plt.imshow(img_back, cmap='gray')
    plt.title('FFT applied filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(f_mag, cmap='gray')
    plt.title('Magnitude spectrum')
    plt.xticks([]), plt.yticks([])
    plt.show() 
    '''
    cv.imwrite(f'{out_dir}/{img_name}.{img_ext}', img_back)

    if gen_spec:
        cv.imwrite(f'{out_dir}/{img_name}_spec.{img_ext}', mag)
        cv.imwrite(f'{out_dir}/{img_name}_spec_cropped.{img_ext}', f_mag)
if __name__ == "__main__":
    main(sys.argv[1:])
