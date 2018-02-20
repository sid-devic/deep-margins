import tensorflow as tf
import numpy as np
from input_pipe import tr_data, val_data

EPOCHS = 10
BATCH_SIZE = 16
# using two numpy arrays

iter = tr_data.make_one_shot_iterator()

x, y = iter.get_next()

# make a simple model
net = tf.layers.dense(x, 16384, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 2, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label

train_op = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))

session = tf.Session()
