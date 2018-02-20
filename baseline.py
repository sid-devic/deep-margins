import tensorflow as tf
import numpy as np
from input_pipe import tr_data, val_data

BATCH_SIZE = 32
img_size = 128
num_channels = 3
num_classes = 2


# tr_data = tr_data.repeat().batch(BATCH_SIZE)

# now we create our iterator
iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)


x, y = iterator.get_next()


# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

# conv_net model
'''
net = tf.layers.dense(x, 16384, activation=tf.tanh)
net = tf.layers.dense(net, 16384, activation=tf.tanh)
prediction = tf.layers.dense(net, 2, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, y)
train_op = tf.train.AdamOptimizer().minimize(loss)
'''

# session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# labels
# y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true = y
y_true_cls = tf.argmax(y_true, dimension=1)



#Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 128 #64

# added
filter_size_conv4 = 3
num_filters_conv4 = 128

filter_size_conv5 = 3
num_filters_conv5 = 128

filter_size_conv6 = 3
num_filters_conv6 = 128

filter_size_conv7 = 3
num_filters_conv7 = 128

filter_size_conv8 = 3
num_filters_conv8 = 128

filter_size_conv9 = 3
num_filters_conv9 = 128

filter_size_conv10 = 3
num_filters_conv10 = 128

filter_size_conv11 = 3
num_filters_conv11 = 128

fc1_layer_size = 128 #128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

layer_conv4 = create_convolutional_layer(input=layer_conv3,
	       num_input_channels=num_filters_conv3,
	       conv_filter_size=filter_size_conv4,
	       num_filters=num_filters_conv4)

layer_conv5 = create_convolutional_layer(input=layer_conv4,
	       num_input_channels=num_filters_conv4,
	       conv_filter_size=filter_size_conv5,
	       num_filters=num_filters_conv5)

layer_conv6 = create_convolutional_layer(input=layer_conv5,
	       num_input_channels=num_filters_conv5,
	       conv_filter_size=filter_size_conv6,
	       num_filters=num_filters_conv6)


layer_conv7 = create_convolutional_layer(input=layer_conv6,
	       num_input_channels=num_filters_conv6,
	       conv_filter_size=filter_size_conv7,
	       num_filters=num_filters_conv7)

layer_conv8 = create_convolutional_layer(input=layer_conv7,
	       num_input_channels=num_filters_conv7,
	       conv_filter_size=filter_size_conv8,
	       num_filters=num_filters_conv8)

layer_conv9 = create_convolutional_layer(input=layer_conv8,
	       num_input_channels=num_filters_conv8,
	       conv_filter_size=filter_size_conv9,
	       num_filters=num_filters_conv9)

layer_conv10 = create_convolutional_layer(input=layer_conv9,
	       num_input_channels=num_filters_conv9,
	       conv_filter_size=filter_size_conv10,
	       num_filters=num_filters_conv10)

layer_conv11 = create_convolutional_layer(input=layer_conv10,
	       num_input_channels=num_filters_conv10,
	       conv_filter_size=filter_size_conv11,
	       num_filters=num_filters_conv11)

layer_flat = create_flatten_layer(layer_conv11)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc1_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc1_layer_size,
                     num_outputs=num_classes,
                     use_relu=True)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)  #1e-4
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# end_model


with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(training_init_op)
        for i in range (10000):
                _, loss_value = sess.run([optimizer, loss])
                print("Iter: {}, loss {:.4f}".format(i, loss_value))



