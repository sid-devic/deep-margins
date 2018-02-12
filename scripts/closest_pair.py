import numpy as np
import cv2
import os
import time

start = time.time()
DIR_PATH ="/home/sid/deep-margins/tutorial/training_data"
image_size = 128

min_cat_img = DIR_PATH + "/cats/2577.jpg"
min_dog_img = DIR_PATH + "/dogs/976.jpg"

i1 = cv2.imread(min_cat_img)
i2 = cv2.imread(min_dog_img)
num_updated = 0

def dist(i1,i2):
    # we crop the image the same way it's done in train.py
    i1 = cv2.resize(i1, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    i2 = cv2.resize(i2, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    #print(np.linalg.norm(i1-i2))
    return np.sqrt(np.sum((i1-i2)**2))

# we don't know scale of distance, so we set a random distance as the min 
min_dist = dist(i1,i2)

for cat_img_path in os.listdir(DIR_PATH + "/cats/"):
    img_1 = cv2.imread(DIR_PATH + "/cats/" + cat_img_path)
    print("Cat img: " + cat_img_path)
    for dog_img_path in os.listdir(DIR_PATH + "/dogs/"):
	img_2 = cv2.imread(DIR_PATH + "/dogs/" + dog_img_path)
        if img_1 is None or img_2 is None:
            continue
        #print(dist(img_1, img_2))
	if dist(img_1, img_2) < min_dist:
            print("min_dist updated:")
            num_updated += 1
            min_dist = dist(img_1, img_2)
	    min_cat_img = DIR_PATH + "/cats/" + cat_img_path
	    min_dog_img = DIR_PATH + "/dogs/" + dog_img_path
            print(min_dist)
            print("Cat img: " + cat_img_path + ", Dog img: " + dog_img_path)

end = time.time()
# output results
print("Took " + str(end-start) + " seconds")
print("Minimum eucl. dist: " + str(min_dist))
print(min_cat_img)
print(min_dog_img)
print("Num updated: " + str(num_updated))
