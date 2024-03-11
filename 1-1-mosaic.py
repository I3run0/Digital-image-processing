import sys
import cv2 as cv
import numpy as np

GRID = 4
MAP = np.array([6, 11, 13, 3, 8, 16, 1, 9, 12, 14, 2, 7, 4, 15, 10, 5]) - 1

def main():
    if len(sys.argv) < 3: 
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])
    
    if img is None:
        sys.exit("Could not read the image")

    new_img = np.empty_like(img)
    img_dim = img.shape 
    img_grid_width = img_dim[1] // GRID
    img_grid_height = img_dim[0] // GRID 

    for i in range(len(MAP)):
        y_init = img_grid_height * (i // 4)
        x_init = img_grid_width * (i % 4)
        y_end = img_grid_height * (i // 4 + 1)
        x_end = img_grid_width * (i % 4 + 1)

        new_y_init = img_grid_height * (MAP[i] // 4)
        new_x_init = img_grid_width * (MAP[i] % 4)
        new_y_end = img_grid_height * (MAP[i] // 4 + 1)
        new_x_end = img_grid_width * (MAP[i] % 4 + 1)

        new_img[y_init:y_end, x_init:x_end, ] = img[new_y_init:new_y_end, new_x_init:new_x_end]

    cv.imwrite(sys.argv[2],new_img)


if __name__ == "__main__":
    main()