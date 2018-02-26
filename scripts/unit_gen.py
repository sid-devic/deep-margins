import cv2
import random
import numpy as np
from pylab import imshow, show, get_cmap
from numpy import *
import os
from PIL import Image
import time

# GLOBAL VARS --

# We generate a preset number of "Random Unit Images" for each item
# in our training set
height = 128
width = 128
gen_per_img = 5
DIR_PATH = "/home/sid/deep-margins/tutorial/training_data/"
UNIT_DATA_PATH = "/home/sid/deep-margins/tutorial/unit_data/"

## Normalization helper func
def find_magnitude(arr):
	running_sum = 0
	for x in np.nditer(arr):
		running_sum = running_sum + (x ** 2)
	return sqrt(running_sum)

def loop_dir(path, extension, save_path):
	path = path + extension
	_, _, file_num = os.walk(path).next()
	file_num = len(file_num)
	count = 0	
	for img in os.listdir(path):
		# Check if valid img
		checked_img = cv2.imread(path + img)
		if checked_img is None:
			continue
		height, width = checked_img.shape[:2]	
		start = time.time()
		count += 1
		for x in range(gen_per_img):
			# We create a random img with the dimensions of loaded img
			imarray = np.random.rand(height, width, 3) * 255
			# we fro norm the image
			imarray = imarray / (find_magnitude(imarray))
			gen_img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
			read_again = asarray(gen_img)
			result_img = Image.fromarray(read_again)
			# now we save our generated image
			result_img.save(save_path + extension + img[:-4] + "_" + str(x) + ".jpg")
		end = time.time()
		print(end-start)
		# fix
		print(str(count) + "/" + str(file_num))

	return 0

print('Creating Cat imgs')
c = loop_dir(DIR_PATH, "cats/", UNIT_DATA_PATH)
print('Creating Dog imgs')
d = loop_dir(DIR_PATH, "dogs/", UNIT_DATA_PATH)

