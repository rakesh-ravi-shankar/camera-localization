import numpy as np
import cv2
from matplotlib import pyplot as plt

def find_pattern(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)

	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth

KNOWN_WIDTH = 11.0

IMAGE_PATHS = ["ImageAssets/IMG_6721.JPG", "ImageAssets/IMG_6722.JPG", "ImageAssets/IMG_6723.JPG"]

focalLength = 29

for imagePath in IMAGE_PATHS:
	image = cv2.imread(imagePath)
	marker = find_pattern(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

	box = np.int0(cv2.boxPoints(marker))

	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	plt.imshow(image)
	plt.show()
