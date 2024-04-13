import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_fft_from_img(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    return f_shift

def get_mag_from_fft(fftshift):
    return 20 * np.log(np.abs(fftshift))

def get_img_from_fft(fftshift):
    f_ishift = np.fft.ifftshift(fftshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return img_back

#------------------------------------------

def get_circle_mask(r, cords, shape, fill):
    mask =  np.zeros(shape, dtype='uint8') if fill\
            else np.ones(shape, dtype='uint8')
    mask = cv.circle(mask, cords, r,\
            1 if fill else 0 , -1)
    return mask

#------------------------------------------

def lower_pass_filter(fft, r):
    mask = get_circle_mask(r, (fft.shape[0]//2, fft.shape[1]//2),\
            fft.shape, True)
    f_fft = fft * mask
    return f_fft

def higher_pass_filter(fft, r):
    mask = get_circle_mask(r, (fft.shape[0]//2, fft.shape[1]//2),\
            fft.shape, False)
    f_fft = fft * mask
    return f_fft

def band_pass_filter(fft, r_higher, r_lower):
    f_fft = higher_pass_filter(fft, r_higher)
    f_fft = lower_pass_filter(f_fft, r_lower)
    return f_fft

def reject_band_filter(fft, r_higher, r_lower):
    f_fft = higher_pass_filter(fft, r_higher)
    mask = get_circle_mask(r_lower, (fft.shape[0]//2, fft.shape[1]//2),\
            fft.shape, True)
    f_fft[mask == 1] = fft[mask == 1]
    return f_fft

def main(argv):
    method = int(argv[0])
    img = cv.imread(argv[1], cv.IMREAD_GRAYSCALE)
    output = 0
    
    f_fft, f_mag = None, None
    
    fft = get_fft_from_img(img)
    mag = get_mag_from_fft(fft)
    if method == 0:
        f_fft = lower_pass_filter(fft, 80)
        f_mag = lower_pass_filter(mag, 80)
    elif method == 1:
        f_fft = higher_pass_filter(fft, 40)
        f_mag = higher_pass_filter(mag, 80)
    elif method == 2:
        f_fft = band_pass_filter(fft, 20, 80)
        f_mag = band_pass_filter(mag, 20, 80)
    elif method == 3:
        f_fft = reject_band_filter(fft, 80, 20)
        f_mag = reject_band_filter(mag, 80, 20)
    
    img_back = get_img_from_fft(f_fft)
    plt.subplot(121)
    plt.imshow(img_back, cmap='grey')
    plt.title('FFT aplied filter')
    plt.xticks([]), plt.yticks([])
    plt.show() 
        
    

if __name__ == "__main__":
    main(sys.argv[1:])
