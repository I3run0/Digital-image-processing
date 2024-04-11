import cv2
import numpy as np

path = r'imagens_png/baboon.png' 
# now we will be loading the image and converting it to grayscale
image = cv2.imread(path)
 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Compute the discrete Fourier Transform of the image
fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
 
# Shift the zero-frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier)
 
# calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
 
# Scale the magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

image_center = (image.shape[0] // 2, image.shape[1] // 2)
radius = 100
mask = np.ones_like(magnitude)
mask = cv2.circle(mask, image_center, radius, (1, 1, 1), -1)
#fourier_shift[:,:,0] *= mask
#fourier_shift[:,:,1] *= mask
mag_mask = magnitude * mask

f_ishift = np.fft.ifftshift(fourier_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Display the magnitude of the Fourier Transform
cv2.imshow('Fourier Transform', magnitude)
cv2.imshow("Image processed", img_back)
#cv2.imshow("Mask applied", mask[:, :, 0])
cv2.imshow("Fourier Transform applied mask", mag_mask)
cv2.waitKey(0)

cv2.destroyAllWindows()
