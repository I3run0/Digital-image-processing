import numpy as np
import cv2
import argparse

def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_nmse(imageA, imageB):
    nmse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) / np.sum(imageA.astype("float")**2)
    return nmse

def main():
    parser = argparse.ArgumentParser(description="Calculate MSE and SSIM between two images.")
    parser.add_argument("imageA", help="Path to the first image.")
    parser.add_argument("imageB", help="Path to the second image.")
    args = parser.parse_args()

    # Carregar imagens
    imageA = cv2.imread(args.imageA, cv2.IMREAD_GRAYSCALE)
    imageB = cv2.imread(args.imageB, cv2.IMREAD_GRAYSCALE)

    if imageA is None or imageB is None:
        print("Error: One of the image paths is invalid.")
        return

    # Calcular MSE e SSIM
    mse_value = calculate_mse(imageA, imageB)
    nmse_value = calculate_nmse(imageA, imageB)

    print(f"MSE: {mse_value}")
    print(f"NMSE: {nmse_value}")

if __name__ == "__main__":
    main()
