import tensorflow as tf
import numpy as np
import random
import cv2
import time

# 12500 imgs of cats, 12500 imgs of dogs

# Home image directory
DIR_PATH="/home/sid/datasets/PetImages"

# our placeholder arrays, which we will turn into actual data later
train_imgs_arr = []
train_labels_arr = []
val_imgs_arr = []
val_labels_arr = []

def input_function(img_path, label):
    """ Function we call on our tf.dataset, takes in 
    image paths and loads images, all natively tf.

    returns: tuple (img, one_hot label vector)
    """
    # label to one-hot
    one_hot = tf.one_hot(label, 2)
    
    #path->image
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file)

    return img_decoded, one_hot

def is_invalid_img(img_path):
    """ Helper function so that we don't load invalid images into our dataset
    """
    checked_img = cv2.imread(DIR_PATH + img_path)
    invalid = False
    if checked_img is None:
        invalid = True   
    return invalid

start = time.time()

# because tf.shuffle(buffer_size) has some issues, we 
# randomly iterate through our data ourself, to make sure it's not ordered.
r = list(range(25000))
random.shuffle(r)
for x in r:
    curr_dir = "/Cat/"
    # we have a "new_x" that we need to keep track of, in case our x is in the
    # range of the /Dog/ dir.
    new_x = x
    # 0 for Cat, 1 for Dog
    img_class = 0
    if x > 12500:
        curr_dir = "/Dog/"
        new_x = x-12500
        img_class = 1
    # we check if the img is a valid one, don't want bad data
    if is_invalid_img(curr_dir + str(new_x)+".jpg"):
        continue
    # we add data to our training set with 80% probability, otherwise it goes
    # to our val set. This is crude, will change later.
    rand_num = random.random()
    if rand_num < .8:
        train_imgs_arr.append(DIR_PATH + curr_dir + str(new_x)+".jpg")
        train_labels_arr.append(int(img_class))
    else:
        val_imgs_arr.append(DIR_PATH + curr_dir + str(new_x) +".jpg")
        val_labels_arr.append(int(img_class))

# now we add our train and val arrs to actual tf.constants
train_imgs = tf.constant(train_imgs_arr)
train_labels = tf.constant(train_labels_arr)
val_imgs = tf.constant(val_imgs_arr)
val_labels = tf.constant(val_labels_arr)

# now we use tf's new Datasets pipeline to create a tuple of imgs and labels
tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
val_data = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))

print(len(val_labels_arr) + len(train_labels_arr))

# we map our data into a readable tf format with our called function
tr_data = tr_data.map(input_function)
val_data = val_data.map(input_function)

# now we create our iterator
iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)

next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

end = time.time()

with tf.Session() as sess:

    # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    # initialize the iterator on the validation data
    sess.run(validation_init_op)

    # get each element of the validation dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of validation dataset.")
            break


print("It took " + str(end - start) + " seconds to load the data.")

