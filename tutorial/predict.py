import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

batch_size = 1000
image_size=128
num_channels=3
cat_images = [[]]
dog_images = [[]]
TEST_DATA_DIR = "/home/sid/deep-margins/tutorial/training_data/"
num = 0

for file in os.listdir(TEST_DATA_DIR + "/cats/"):
	if len(cat_images[num]) >= batch_size:
		num += 1
		cat_images.append([])
		continue
	filename = TEST_DATA_DIR + "/cats/" + file
	# Reading the image using OpenCV
	image = cv2.imread(filename)
	if image is None:
		continue
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	cat_images[num].append(image)
	print(str(num) + ": " + str(len(cat_images[num])))

num = 0
for file in os.listdir(TEST_DATA_DIR + "/dogs/"):
	if len(dog_images[num]) >= batch_size:
		num += 1
		dog_images.append([])
		continue
	filename = TEST_DATA_DIR + "/dogs/" + file
	# Reading the image using OpenCV
	image = cv2.imread(filename)
	if image is None:
		continue
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	dog_images[num].append(image)
	print(str(num) + ": " + str(len(dog_images[num])))

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('test_model_5460.ckpt.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess,'test_model_5460.ckpt')

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 

cat_results = []
dog_results = []
### Creating the feed_dict that is required to be fed to calculate y_pred 
for i in range(len(cat_images)):
	inputs = cat_images[i]
	inputs = np.array(inputs, dtype=np.uint8)
	inputs = inputs.astype('float32')
	inputs = np.multiply(inputs, 1.0/255.0) 

	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = inputs.reshape(len(inputs), image_size,image_size,num_channels)
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	cat_result=sess.run(y_pred, feed_dict=feed_dict_testing)
	# result is of this format [probabiliy_of_rose probability_of_sunflower]
	cat_results.append(cat_result)

for i in range(len(dog_images)):
	inputs = dog_images[i]
	inputs = np.array(inputs, dtype=np.uint8)
	inputs = inputs.astype('float32')
	inputs = np.multiply(inputs, 1.0/255.0) 

	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = inputs.reshape(len(inputs), image_size,image_size,num_channels)
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	dog_result=sess.run(y_pred, feed_dict=feed_dict_testing)
	# result is of this format [probabiliy_of_rose probability_of_sunflower]
	dog_results.append(dog_result)

cat_results = [j for i in cat_results for j in i]
dog_results = [j for i in dog_results for j in i]

print("Number of cat pics: " + str(len(cat_results)))
print("Number of dog pics: " + str(len(dog_results)))

count = 0
for x in cat_results:
	if x[0] > x[1]:
		count += 1
print("incorrectly classified cats: " + str(count))
for x in dog_results:
	if x[0] < x[1]:
		count += 1

print(count)
