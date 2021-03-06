import cv2
import random
import numpy as np
from pylab import imshow, show, get_cmap
from numpy import *
import os
from PIL import Image
import time
from sklearn.preprocessing import normalize

#-------------------- settings -----------------------##
## Create a set of random images around a given image ##
## Cat img: 4451.jpg, Dog img: 11942.jpg              ## 
## min dist: 15247.1165471                            ##
######################################################## 

min_dist = 40000
height = 128
width = 128
gen_per_img = 2
DIR_PATH = "/home/sid/deep-margins/tutorial/training_data/"
GEN_DIR_PATH = "/home/sid/deep-margins/tutorial/generated_data/"



# keep track of how many images we've augmented
path_cat, dirs_cat, files_cat = os.walk(DIR_PATH + "cats/").next()
path_dog, dirs_dog, files_dog = os.walk(DIR_PATH + "dogs/").next()
file_count = len(files_cat + files_dog)
count = 0

def find_magnitude(arr):
    running_sum = 0
    for x in np.nditer(arr):
        running_sum = running_sum + (x ** 2)
    return sqrt(running_sum)

# loop over all cat images
for cat_img in os.listdir(DIR_PATH + "cats/"):
    # check if the image is valid
    checked_img = cv2.imread(DIR_PATH + "cats/" + cat_img)
    if checked_img is None:
        continue
    print(cat_img)
    start = time.time()
    count += 1
    # we generate gen_per_img images per each image
    for x in range(gen_per_img):
        # load in the image
        img = Image.open(DIR_PATH + "cats/" + cat_img)
        width, height = img.size
        # we create a random image with the dimensions of the loaded image
        imarray = np.random.rand(height, width, 3) * 255
        # we normalize the image (corresponding 'unit' image)
        #img_min = imarray.min(axis=(1, 2), keepdims=True)
        #img_max = imarray.max(axis=(1, 2), keepdims=True)
        #imarray = (imarray - img_min)/(img_max - img_min)

        # we 'scale' our unit image by a factor of half the minimum distance 
        # between a cat and dog image (our least 'margin'), after we normalize it
        imarray = imarray / (find_magnitude(imarray))
	imarray = imarray * min_dist / 2
        print(imarray)
	# we convert our nparray into a proper 3 channel img
        gen_img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        # we load both images back into an array
        im1arr = asarray(img)
        im2arr = asarray(gen_img)
        # add the images together. This is the same as moving the normalized
        # vector to the area of space where our reference image resides
        if im1arr.size != im2arr.size:
            continue
        addition = im1arr + im2arr
        result_img = Image.fromarray(addition)
        # finally, we save our generated image
        tmp_DIR_PATH = GEN_DIR_PATH + "cats/" + cat_img
        tmp_DIR_PATH = tmp_DIR_PATH[:-4]
        tmp_DIR_PATH += "_" + str(x) + ".jpg"
        result_img.save(tmp_DIR_PATH)
    
    end = time.time()        
    print(end-start)
    print(str(count) + "/" + str(file_count))

# loop over all dog images
for dog_img in os.listdir(DIR_PATH + "dogs/"):
    # check if the image is valid, if invalid we skip it
    checked_img = cv2.imread(DIR_PATH + "dogs/" + dog_img)
    if checked_img is None:
        continue
    # debug/logs
    print(dog_img)
    start = time.time()
    count += 1
    # we generate gen_per_img images per each image
    for x in range(gen_per_img):
        # load in the image
        img = Image.open(DIR_PATH + "dogs/" + dog_img)
        width, height = img.size
        # we create a random image with the dimensions of the loaded image
        imarray = np.random.rand(height, width, 3) * 255
        # we normalize the image (corresponding 'unit' image)
        #img_min = imarray.min(axis=(1, 2), keepdims=True)
        #img_max = imarray.max(axis=(1, 2), keepdims=True)
        #imarray = (imarray - img_min)/(img_max - img_min)
	
        # we 'scale' our unit image by a factor of half the minimum distance 
        # between a cat and dog image (our least 'margin'), after we normalize it
        imarray = imarray / (find_magnitude(imarray))
	imarray = imarray * min_dist / 2
        # we convert our nparray into a proper 3 channel img
        gen_img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        # we load both images back into an array
        im1arr = asarray(img)
        im2arr = asarray(gen_img)
        # add the images together. This is the same as moving the normalized
        # vector to the area of space where our reference image resides
        if im1arr.size != im2arr.size:
            continue
        addition = im1arr + im2arr
        result_img = Image.fromarray(addition)
        # finally, we save our generated image
        tmp_DIR_PATH = GEN_DIR_PATH + "dogs/" + dog_img
        tmp_DIR_PATH = tmp_DIR_PATH[:-4]
        tmp_DIR_PATH += "_" + str(x) + ".jpg"
        result_img.save(tmp_DIR_PATH)
    
    end = time.time()        
    print(end-start)
    print(str(count) + "/" + str(file_count))
