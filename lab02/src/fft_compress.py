import sys
import numpy as np
import cv2 as cv
import fft_utils as uts


def main(argv):
    if len(argv) < 3:
        print("Usage message: to do")
        sys.exit()

    cmps_level = float(argv[0])
    if cmps_level < 1:
        print("Usafge message: to do")

    img = cv.imread(argv[1], cv.IMREAD_GRAYSCALE)
    out_p = argv[2]
    
    fft = uts.get_fft_from_img(img)

    sl_fft = np.reshape(fft, (img.shape[0] * img.shape[1]))
    sl_fft.sort()
    sl_fft = sl_fft[::-1]
    sl_fft = sl_fft[:round(cmps_level * sl_fft.shape[0])]
    threshold = sl_fft[-1]

    fft[fft < threshold] = 0
    
    img_back = uts.get_img_from_fft(fft)
    
    cv.imwrite(out_p, img_back)







if __name__ == '__main__':
    main(sys.argv[1:])
