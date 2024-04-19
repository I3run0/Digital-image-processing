import numpy as np

def get_fft_from_img(img):
    """
    Compute the Fourier Transform of an image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Shifted Fourier Transform of the input image.
    """
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    return fft_shift

def get_mag_from_fft(fft_shift):
    """
    Compute the magnitude spectrum from the shifted Fourier Transform.

    Args:
        fft_shift (numpy.ndarray): Shifted Fourier Transform of the image.

    Returns:
        numpy.ndarray: Magnitude spectrum.
    """
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
    return magnitude_spectrum

def get_img_from_fft(fft_shift):
    """
    Compute the inverse Fourier Transform to reconstruct the image.

    Args:
        fft_shift (numpy.ndarray): Shifted Fourier Transform.

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    f_ishift = np.fft.ifftshift(fft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)  # Take the real part to remove imaginary artifacts
    return img_back
<<<<<<< HEAD


=======
>>>>>>> 5ebcc7939279a90cd9c61c6898bd2481dc67c411
