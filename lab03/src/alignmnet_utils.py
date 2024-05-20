
import cv2 as cv
import numpy as np

def rotate_image(img, angle):
	height, width = img.shape[:2]
	new_height = int(width * np.abs(np.sin(np.radians(angle))) + height * np.abs(np.cos(np.radians(angle))))
	new_width = int(height * np.abs(np.sin(np.radians(angle))) + width * np.abs(np.cos(np.radians(angle))))
	rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
	rotation_matrix[0, 2] += (new_width - width) / 2
	rotation_matrix[1, 2] += (new_height - height) / 2
	rotated_image = cv.warpAffine(img, rotation_matrix, (new_width, new_height), flags=cv.INTER_LINEAR)
	return rotated_image