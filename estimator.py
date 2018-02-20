import tensorflow as tf
import numpy as np
from input_pipe import tr_data, val_data
import image

# Parameters
learning_rate = 1e-4
num_epochs = 2000
batch_size = 1
display_step = 100

# We know that the images are 128x128
img_size = 128

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Number of classes, Dogs and Cats
num_classes = 2

# We import our Dataset objects from input_pipe because we need to be
# able to access them from within our input functions
train_dataset = tr_data
val_dataset = val_data

def train_input_fn():
	tr_data = train_dataset.batch(batch_size)
	#tr_data = tr_data.repeat(num_epochs)
	#tr_data = tr_data.repeat()
	iterator = train_dataset.make_one_shot_iterator()
	# iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)

	features, labels = iterator.get_next()
	x = {'image': features}
	y = labels
	return x, y

def test_input_fn():
	#iterator = val_dataset.make_one_shot_iterator()
	#iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)

	features, labels = iterator.get_next()

	return features, labels


feature_image = tf.feature_column.numeric_column('image', shape=[16384*3], dtype=tf.float32)
num_hidden_units = [512, 256, 128]
model = tf.estimator.DNNClassifier(feature_columns=[feature_image],
				hidden_units=num_hidden_units,
				activation_fn=tf.nn.relu,
				n_classes=num_classes,
				model_dir="./checkpoints")

model.train(input_fn=train_input_fn, steps=100)
