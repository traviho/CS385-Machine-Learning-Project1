import cv2
import numpy as np
import math

#### Cropping functions ####

BOUNDING_BOX_FACTOR = 4/3
RESIZE_IMG_SIZE = (96, 96)

def crop_image(img, x1, y1, x2, y2):
	# (x1, y1) is top-left corner of where image should be cropped, relative to (0, 0). (x2, y2) is right-bottom corner.
	# if rectangle is out of bounds of normal image, we add edge-padding
	if (x1 < 0 or x2 > img.shape[1] or y1 < 0 or y2 > img.shape[0]):
		# pad_size is size of largest distance crop box is out of bounds by
		left_diff = abs(0 - x1) if x1 < 0 else 0
		top_diff = abs(0 - y1) if y1 < 0 else 0
		right_diff = abs(x2 - img.shape[1]) if x2 > img.shape[1] else 0
		bottom_diff = abs(y2 - img.shape[0]) if y2 > img.shape[0] else 0
		pad_size = max(left_diff, top_diff, right_diff, bottom_diff)

		edge_padded = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
		# top-left corner is moved
		x1 = x1 + left_diff 
		y1 = y1 + top_diff

		return edge_padded[y1:y2, x1:x2] 
	return img[y1:y2, x1:x2] # image subset

def bounding_box(center_x, center_y, radius_x, radius_y, with_factor=True):
	bounding_factor = BOUNDING_BOX_FACTOR if with_factor else 1
	x1, y1 = math.floor(center_x - (bounding_factor * radius_x)), math.floor(center_y - (bounding_factor * radius_y))
	x2, y2 = math.floor(center_x + (bounding_factor * radius_x)), math.floor(center_y + (bounding_factor * radius_y))
	return ((x1, y1), (x2, y2))

def resize_img_with_values(img, center_x, center_y, radius_x, radius_y):
	resized_img = cv2.resize(img, RESIZE_IMG_SIZE)
	x_ratio = RESIZE_IMG_SIZE[0] / img.shape[1]
	y_ratio = RESIZE_IMG_SIZE[1] / img.shape[0]
	resized_center_x = math.floor(x_ratio * center_x)
	resized_center_y = math.floor(y_ratio * center_y)
	resized_radius_x = math.floor(x_ratio * radius_x)
	resized_radius_y = math.floor(y_ratio * radius_y)
	return (resized_img, resized_center_x, resized_center_y, resized_radius_x, resized_radius_y)

def gen_postive_img(img, center_x, center_y, radius_x, radius_y):
	(x1, y1), (x2, y2) = bounding_box(center_x, center_y, radius_x, radius_y)
	return crop_image(img, x1, y1, x2, y2)

# returns 8 images sliding around bounding box
def sliding_boxes(img, radius_x, radius_y):
	box_width = radius_x * 2 # width & height of original bounding box, without the bounding box factor
	box_height = radius_x * 2
	x1, y1 = (0, 0)
	x2, y2 = (img.shape[1], img.shape[0])
	center_x = math.floor(x2 / 2)
	center_y = math.floor(y2 / 2)
	unbounded_x1, unbounded_y1 = center_x - radius_x, center_y - radius_y # x1, y1 point of inner-bounding box (without factor)

	neg_1 = img[y1:y1 + box_height, x1: x1 + box_width]
	neg_2 = img[y1:y1 + box_height, unbounded_x1: unbounded_x1 + box_width]
	neg_3 = img[y1:y1 + box_height, x2 - box_width: x2]

	neg_4 = img[unbounded_y1:unbounded_y1 + box_height, x1:x1 + box_width]
	neg_5 = img[unbounded_y1:unbounded_y1 + box_height, x2 - box_width: x2]

	neg_6 = img[y2 - box_height:y2, x1:x1 + box_width]
	neg_7 = img[y2 - box_height:y2, unbounded_x1: unbounded_x1 + box_width]
	neg_8 = img[y2 - box_height:y2, x2 - box_width: x2]
	return [neg_1, neg_2, neg_3, neg_4, neg_5, neg_6, neg_7, neg_8]

def gen_negative_images(img, center_x, center_y, radius_x, radius_y):
	resized_img, resized_center_x, resized_center_y, resized_radius_x, resized_radius_y = resize_img_with_values(img, center_x, center_y, radius_x, radius_y)
	(x1, y1), (x2, y2) = bounding_box(resized_center_x, resized_center_y, resized_radius_x, resized_radius_y)
	cropped_img = crop_image(resized_img, x1, y1, x2, y2)
	return sliding_boxes(cropped_img, resized_radius_x, resized_radius_y)

def gen_data_with_labels(face_data_arr, gen_negatives=False):
	X_data = []
	Y_data = []
	for face_data in face_data_arr:
		X_data.append(gen_postive_img(*face_data))
		Y_data.append(1)
		if (gen_negatives):
			X_data += gen_negative_images(*face_data)
			Y_data += [-1] * 8
	return (X_data, Y_data)

#### File I/O functions ####

IMAGE_PATH = 'originalPics/'
FOLDER_PATH = 'FDDB-folds/'

# given ellipse text file name, location of images, returns [face data] where face data = (img, center_x, center_y, radius_x, radius_y)
def read_ellipse_text(text_file_name, image_path):
	face_data_arr = []
	text_file = open(text_file_name, "r")
	lines = text_file.read().split('\n')[0:-1]
	i = 0
	img_name = ""
	while (i < len(lines)):
		if ("img" in lines[i]):
			img_name = lines[i].strip('\n')
			img_matrix = cv2.imread(image_path + img_name + ".jpg")
			num_faces = int(lines[i + 1].strip('\n'))
			i += 2
			for j in range(i, i + num_faces):
				face_data = lines[j].split(" ") # x_rad, y_rad, radians, x, y, 1
				radius_y = float(face_data[0])
				radius_x = float(face_data[1])
				center_x = float(face_data[3])
				center_y = float(face_data[4])
				face_data_arr.append((img_matrix, center_x, center_y, radius_x, radius_y))
			i += num_faces
	return face_data_arr

# given location of text files, images, and range, returns [face data] in range where face data = (img, center_x, center_y, radius_x, radius_y)
def read_ellipse_text_in_range(interval, folder_path, image_path):
	face_data_arr = []
	for i in range(*interval):
		text_file_name = folder_path + "FDDB-fold-0%d-ellipseList.txt" % i
		if i > 9:
			text_file_name = folder_path + "FDDB-fold-%d-ellipseList.txt" % i
		face_data_arr += read_ellipse_text(text_file_name, image_path)
	return face_data_arr

#### Create dataset using cropping + File I/O functions ####

# Arbitrary dataset where first 4 folders (pos + neg), first 8 train, last 2 test, last 1 (pos + neg)
def create_data():
	X_train, Y_train, X_test, Y_test = [], [], [], []
	face_data_arr_1 = read_ellipse_text_in_range((1, 5), FOLDER_PATH, IMAGE_PATH) # positives
	face_data_arr_2 = read_ellipse_text_in_range((5, 9), FOLDER_PATH, IMAGE_PATH) # positives & negatives
	face_data_arr_3 = read_ellipse_text_in_range((9, 10), FOLDER_PATH, IMAGE_PATH) # positives
	face_data_arr_4 = read_ellipse_text_in_range((10, 11), FOLDER_PATH, IMAGE_PATH) # positives & negatives

	# TODO: Adding two tuples (l1, l2) + (l3, l4) -> (l1 + l3, l2 + l4) more concise?
	X_train, Y_train = gen_data_with_labels(face_data_arr_1)
	X_train_add, Y_train_add = gen_data_with_labels(face_data_arr_2, True)
	X_train += X_train_add
	Y_train += Y_train_add
	
	X_test, Y_test = gen_data_with_labels(face_data_arr_3)
	X_test_add, Y_test_add = gen_data_with_labels(face_data_arr_4, True)
	X_test += X_test_add
	Y_test += Y_test_add

	return (X_train, Y_train, X_test, Y_test)

#### Tests ####

# X_train, Y_train, X_test, Y_test = create_data()
# print(len(X_train))
# print(len(Y_train))

# print(len(X_test))
# print(len(Y_test))
# cv2.imshow('image', X_train[8])
# cv2.waitKey(0)