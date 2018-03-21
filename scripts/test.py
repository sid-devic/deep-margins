import cv2
import numpy as np

# 10658.jpg

im_1 = cv2.imread('/home/sid/deep-margins/tutorial/generated_data/cats/10658_0.jpg')
im_2 = cv2.imread('/home/sid/deep-margins/tutorial/training_data/cats/10658.jpg')

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

if im_1 is not im_2:
	print(im_1)
	print('===========================================')
	print(im_2)
