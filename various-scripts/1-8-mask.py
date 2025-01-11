import sys
import cv2 as cv
import numpy as np

h1 = np.array([[0, 0, -1, 0, 0],
               [0, -1, -2, -1, 0],
               [-1, -2, 16, -2, -1],
               [0, -1, -2, -1, 0],
               [0, 0, -1, 0, 0]])

h2 = (1/256) * np.array([[1, 4, 6, 4, 1],
               [4, 16, 24, 16, 4],
               [6, 24, 36, 24, 6],
               [4, 16, 24, 16, 4],
               [1, 4, 6, 4, 1]])

h3 = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])

h4 = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

h5 = np.array([[-1, -1, -1],
               [-1, 8, -1],
               [-1, -1, -1]])

h6 = (1/9) * np.ones((3, 3))

h7 = np.array([[-1, -1, 2],
             [-1, 2, -1],
             [2, -1, -1]])

h8 = np.array([[2, -1, -1],
               [-1, 2, -1],
               [-1, -1, 2]])

h9 = (1/9) * np.identity(9)

h10 = (1/8) * np.array([[-1, -1, -1, -1, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, 2, 8, 2, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, -1, -1, -1, -1]])

h11 = np.array([[-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1]])

filters = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11]

def main():
    if len(sys.argv) < 4:
        sys.exit("Parameters are missing")

    img = cv.imread(sys.argv[1])
    
    if img is None :
        sys.exit("Could not read the images")

    tp = int(sys.argv[3])
    img = cv.filter2D(img, -1, filters[tp - 1])
    cv.imwrite(sys.argv[2], img)


if __name__ == "__main__":
    main()