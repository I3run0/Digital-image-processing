import cv2
import sys
import numpy as np

def apply_low_pass_filter(image, cutoff_frequency):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform 2D FFT
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    # Apply low-pass filter
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
    fshift = fshift * mask

    # Shift frequency spectrum back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

# Load the image
image_path = sys.argv[1]
image = cv2.imread(image_path)

# Set cutoff frequency (adjust as needed)
cutoff_frequency = 50

# Apply low-pass filter
filtered_image = apply_low_pass_filter(image, cutoff_frequency)

# Display original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', np.uint8(filtered_image))
cv2.waitKey(0)
cv2.destroyAllWindows()

