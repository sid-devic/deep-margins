import cv2
import random
import numpy as np
from pylab import imshow, show, get_cmap
from numpy import *
import os
from PIL import Image
import time
import shutil
import os
import sys

min_dist = int(sys.argv[1])

SAVE_PATH = "/home/sid/deep-margins/cifar/generated_data/"
UNIT_PATH = "/home/sid/deep-margins/cifar/unit_data/"
BOUNDARY_UNIT_PATH = "/home/sid/deep-margins/cifar/boundary_unit_data/"
TRAIN_PATH = "/home/sid/deep-margins/cifar/train/"

def loop_dir(path, extension, unit_path, save_path, interior, gen_per_img):
        # only clear the dir if we are generating interior images
        if interior is True:
                # clear generated_data directory
                folder = save_path + extension
                for the_file in os.listdir(folder):
                        file_path = os.path.join(folder, the_file)
                        try:
                                if os.path.isfile(file_path):
                                        os.unlink(file_path)
                                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                        except Exception as e:
                                print(e)

        path = path + extension
        _, _, file_num = os.walk(path).next()
        file_num = len(file_num)
        count = 0
        for img in os.listdir(path):
                start = time.time()
		# Check if valid img
                checked_img = cv2.imread(path + img)
               
                if checked_img is None:
                        continue
                
                checked_img = cv2.cvtColor(checked_img, cv2.COLOR_BGR2RGB)
                img_num = (path + img)[:-4]
                img_num = img_num.replace(path, '')	
                count += 1
                for x in range(gen_per_img):
		        # load and scale our unit image
                        path_to_unit_img = unit_path + extension + img_num + '_' + str(x) + '.npy'
                        unit_img = np.load(path_to_unit_img)

                        # sample uniformly from interior of n-sphere of r = min_dist
                        if interior is True:
                                rand_int = random.randint(1,min_dist)
                        else:
                                rand_int = min_dist
                        
                        unit_img = rand_int * unit_img

			# check if our unit img and our original image are the same size
                        if unit_img.size != checked_img.size:
                                continue
                        # "move" our new image to the solution space of the original img
                        addition = unit_img + checked_img
                        
                        result_img = Image.fromarray(addition.astype('uint8')).convert('RGB')
                        # now we save our generated image
                        if interior is True:
                                result_img.save(save_path + extension + img[:-4] + "_" + str(x) + ".png")
                        else:
                               result_img.save(save_path + extension + img[:-4] + "_" + "b" + "_" + str(x) + ".png")
                end = time.time()
                print(end-start)
                # fix
                print(str(count) + "/" + str(file_num))

        return 0
'''
# generate 20 interior and 10 boundary images per image in our training set
c = loop_dir(TRAIN_PATH, "cats/", UNIT_PATH, SAVE_PATH, True, 0)
d = loop_dir(TRAIN_PATH, "dogs/", UNIT_PATH, SAVE_PATH, True, 0)
'''
e = loop_dir(TRAIN_PATH, "cats/", UNIT_PATH, SAVE_PATH, False, 20)
f = loop_dir(TRAIN_PATH, "dogs/", UNIT_PATH, SAVE_PATH, False, 20)

print(min_dist)
