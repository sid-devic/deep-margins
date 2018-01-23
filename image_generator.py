import cv2
import random
import numpy as np
from pylab import imshow, show, get_cmap
from numpy import *
import os
from PIL import Image
import time
from sklearn.preprocessing import normalize

## Create a set of random images around a given image
## Cat img: 4451.jpg, Dog img: 11942.jpg
## min dist: 15247.1165471
min_dist = 15247.1165
height = 128
width = 128
gen_per_img = 33
DIR_PATH = "/home/sid/tensorflow2.7/projects/tutorials/Tensorflow-tutorials/tutorial-2-image-classifier/training_data/"

'''
for x in range(10):
    gen_img = np.random.random_sample((height, width,3))
    cv2.imwrite(DIR_PATH + "cats/generated_images/" + str(x)+".jpg", gen_img)
'''
path, dirs, files = os.walk(DIR_PATH + "cats/").next()
file_count = len(files)
count = 0

# loop over all images
for cat_img in os.listdir(DIR_PATH + "cats/"):
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
        img_min = imarray.min(axis=(1, 2), keepdims=True)
        img_max = imarray.max(axis=(1, 2), keepdims=True)
        imarray = (imarray - img_min)/(img_max - img_min)
        # we 'scale' our unit image by a factor of half the minimum distance 
        # between a cat and dog image (our least 'margin')
        imarray = imarray * min_dist / 2
        # we convert our nparray into a proper 3 channel img
        gen_img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        # we load both images back into an array
        im1arr = asarray(img)
        im2arr = asarray(gen_img)
        # add the images together. This is the same as moving the normalized
        # vector to the area of space where our reference image resides
        addition = im1arr + im2arr
        result_img = Image.fromarray(addition)
        # finally, we save our generated image
        tmp_DIR_PATH = DIR_PATH + "cats/generated_images/" + cat_img
        tmp_DIR_PATH = tmp_DIR_PATH[:-4]
        tmp_DIR_PATH += "_" + str(x) + ".jpg"
        result_img.save(tmp_DIR_PATH)
    
    end = time.time()        
    print(end-start)
    print(str(count) + "/" + str(file_count))
