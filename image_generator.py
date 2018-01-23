import cv2
import random
import numpy as np
from pylab import imshow, show, get_cmap
from numpy import *
import os
from PIL import Image
import time

## Create a set of random images around a given image
## Cat img: 4451.jpg, Dog img: 11942.jpg
## min dist: 15247.1165471
height = 128
width = 128
gen_per_img = 33
DIR_PATH = "/home/sid/tensorflow2.7/projects/tutorials/Tensorflow-tutorials/tutorial-2-image-classifier/training_data/"

'''
for x in range(10):
    gen_img = np.random.random_sample((height, width,3))
    cv2.imwrite(DIR_PATH + "cats/generated_images/" + str(x)+".jpg", gen_img)
'''

for cat_img in os.listdir(DIR_PATH + "cats/"):
    print(cat_img)
    start = time.time()
 
    for x in range(gen_per_img):
        img = Image.open(DIR_PATH + "cats/" + cat_img)
        width, height = img.size
        imarray = np.random.rand(height, width, 3) * 255
        rand_img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        im1arr = asarray(img)
        im2arr = asarray(rand_img)
        
        addition = im1arr + im2arr
        result_img = Image.fromarray(addition)
        tmp_DIR_PATH = DIR_PATH + "cats/generated_images/" + cat_img
        tmp_DIR_PATH = tmp_DIR_PATH[:-4]
        tmp_DIR_PATH += "_" + str(x) + ".jpg"
        result_img.save(tmp_DIR_PATH)
        end = time.time()
        
    print(end-start)
