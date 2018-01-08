import cv2
import numpy as np
import scipy.misc

# Crops all Cat and Dog images into 128x128x0 (greyscale)
DIR_PATH = "/home/sid/datasets/PetImages"

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

for x in range(0,12500):
	cropped_img = cv2.imread(DIR_PATH + "/Cat/"+str(x)+".jpg")
	if cropped_img is not None:
		cropped_img = crop_center(cropped_img, 128, 128)
		cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
		scipy.misc.imsave(DIR_PATH + "/Cat/"+str(x)+".jpg", cropped_img)

for x in range(0,12500):
        cropped_img = cv2.imread(DIR_PATH + "/Dog/"+str(x)+".jpg")
        if cropped_img is not None:
                cropped_img = crop_center(cropped_img, 128, 128)
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                scipy.misc.imsave(DIR_PATH + "/Dog/"+str(x)+".jpg", cropped_img)

