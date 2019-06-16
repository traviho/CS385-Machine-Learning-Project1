import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from get_image_data import get_image_data, bounding_box, read_ellipse_text_in_range

NUMBER_OF_BINS = 9
CELL_SIZE = (16, 16)
BLOCK_SIZE = (2, 2)

def visualize_HOG(image):
	_, hog_image = hog(image, orientations=NUMBER_OF_BINS, pixels_per_cell=CELL_SIZE,
                    cells_per_block=BLOCK_SIZE, visualize=True, multichannel=True)
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	plt.show()

def visualize_HOG_data():
	X_train, _, X_test, _ = get_image_data()

	X_train = X_train[:2] # minimize number shown
	X_test = X_test[:1] # minimize number shown
	for image in X_train:
		visualize_HOG(image)

	for image in X_test:
		visualize_HOG(image)


IMAGE_PATH = 'originalPics/'
FOLDER_PATH = 'FDDB-folds/'
BOUNDING_BOX_FACTOR = 4/3

def visualize_bounding_box(img, center_x, center_y, radius_x, radius_y):
	p1, p2 = bounding_box(center_x, center_y, radius_x, radius_y)
	img = cv2.rectangle(img, p1, p2, (0,255,0), 3)
	cv2.imshow("img", img)
	cv2.waitKey(0)

def visualize_bounding_box_data():
	face_data_arr = read_ellipse_text_in_range((1, 11), FOLDER_PATH, IMAGE_PATH)
	face_data_arr = face_data_arr[:4] # minimize number shown
	for face_data in face_data_arr:
		visualize_bounding_box(*face_data)

# visualize_HOG_data()
# visualize_bounding_box_data()