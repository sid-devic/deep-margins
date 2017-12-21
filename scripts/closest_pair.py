import numpy as np
import cv2


i1 = cv2.imread("PetImages/Cat/99.jpg")
i2 = cv2.imread("PetImages/Dog/9.jpg")

def dist(i1,i2):
	return np.linalg.norm(i1-i2)

# we don't know scale of distance, so we set a random distance as the min 
min = dist(i1,i2)

for x in range(0,12500):
        img_1 = cv2.imread("PetImages/Cat/"+str(x)+".jpg")
	for y in range(0,12500):
		img_2 = cv2.imread("PetImages/Dog"+str(y)+".jpg")
		if img_1 is not None and img_2 is not None:
			if dist(img_1, img_2) < min:
				min = dist(img_1, img_2)
				print(img_1.path)
				print(min)

