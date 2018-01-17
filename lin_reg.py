import tensorflow as tf
import numpy as np
from input_pipe import tr_data, train_imgs, train_labels

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 128*128])
y_ = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.zeros([128*128, 2]))
b = tf.Variable(tf.zeros[2])

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(tf.reduce_sum(y * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

dataset = tr_data.batch(128)
iterator = dataset.make_initializable_iterator()
train_step.run(iterator)
print(accuracy)

# github test
