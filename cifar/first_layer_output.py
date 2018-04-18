import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
from random import randint
import matplotlib.pyplot as plt

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# iterate over everything in our train set
batch_size = 1
restore_model_id = 1260

#Prepare input data
classes = ['dogs','cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
# We've modified this so it pulls from training and testing_data respectively
validation_size = 0
img_size = 32
num_channels = 3

# If train_path set to generated_data, we are not training on the original data, we are training
# on generated fake data.
train_path="train"

val_path = "test"

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, val_path, img_size, classes)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32


# "unrolling" first layer so we can access individual variables

layer_conv1_weights = tf.Variable(tf.truncated_normal(name="a", 
                                        shape=[filter_size_conv1, filter_size_conv1, num_channels, num_filters_conv1], 
                                        stddev=0.05))
layer_conv1_bias = tf.Variable(tf.constant(name="b", value=0.05, shape=[num_filters_conv1]))

out_layer_conv1 = tf.nn.conv2d(input=x,
                     filter=layer_conv1_weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
out_layer_conv1 += layer_conv1_bias
out_layer_conv1 = tf.nn.max_pool(value=out_layer_conv1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
layer_conv1 = tf.nn.relu(out_layer_conv1)

print(layer_conv1.get_shape())

#session.run(tf.global_variables_initializer()) 


total_iterations = 0

saver = tf.train.Saver({layer_conv1_weights, layer_conv1_bias})

saver.restore(session, "models/test_model_"+str(restore_model_id)+".ckpt")

x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)

feed_dict_tr = {x: x_batch, y_true: y_true_batch}

g = session.run(layer_conv1, feed_dict=feed_dict_tr)


def vis_conv(v,ix,iy,ch,cy,cx, p = 0):
    v = np.reshape(v,(iy,ix,ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy,ix,cy,cx))
    v = np.transpose(v,(2,0,3,1))   #cy,iy,cx,ix
    v = np.reshape(v,(cy*iy,cx*ix))
    return v

ix = 16  # data size
iy = 16
ch = 32   
cy = 4   # grid from channels:  32 = 4x8
cx = 8

v  = vis_conv(g,ix,iy,ch,cy,cx)
print(v.shape)
plt.figure(figsize = (8,8))
plt.imshow(v,cmap="Greys_r",interpolation='nearest')
plt.show()