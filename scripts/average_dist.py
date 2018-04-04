import numpy as np
import cv2
import os
import time


start = time.time()
DIR_PATH ="/home/sid/deep-margins/cifar/train"
image_size = 32

min_cat_img = DIR_PATH + "/cats/9_cat.png"
min_dog_img = DIR_PATH + "/dogs/999_dog.png"

i1 = cv2.imread(min_cat_img)
i2 = cv2.imread(min_dog_img)
num_updated = 0

def dist(i1,i2):
    # we crop the image the same way it's done in train.py
    i1 = cv2.resize(i1, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    i2 = cv2.resize(i2, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    #print(np.linalg.norm(i1-i2))
    return np.sqrt(np.sum((i1-i2)**2))

distances = []

_, _, file_num = os.walk(DIR_PATH + "/cats/").next()
file_num = len(file_num)
count = 0
for cat_img_path in os.listdir(DIR_PATH + "/cats/"):
    count += 1
    img_1 = cv2.imread(DIR_PATH + "/cats/" + cat_img_path)
    print("Cat img: " + cat_img_path)
    for dog_img_path in os.listdir(DIR_PATH + "/dogs/"):
	img_2 = cv2.imread(DIR_PATH + "/dogs/" + dog_img_path)
        if img_1 is None or img_2 is None:
            continue
        #print(dist(img_1, img_2))
        distances.append(dist(img_1, img_2))
	
    print(str(count) + "/" + str(file_num))

end = time.time()
print(distances)
average = sum(distances) / float(len(distances))
median = np.median(distances)
percent = np.percentile(distances, 99)
# output results
print("Took " + str(end-start) + " seconds")
print("Average eucl. dist: " + str(average))
print("Median dist: " + str(median))
print("99th Percentile: " + str(percent))

