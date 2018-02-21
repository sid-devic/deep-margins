import tensorflow as tf
import numpy as np
from input_pipe import tr_data, val_data
import image

# Parameters
learning_rate = 1e-4
num_epochs = 20
batch_size = 128
display_step = 100

# We know that the images are 128x128
img_size = 128
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3
# Number of classes: [Cats, Dogs]
num_classes = 2

temp_train = tr_data
temp_val = val_data

def train_input_fn():
	train_dataset = tr_data.batch(batch_size)
	# train_dataset = train_dataset.repeat()
	# train_dataset = train_dataset.shuffle(buffer_size=batch_size)	

	iterator = train_dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	x = {'image': features}
	y = labels

	return x, y

def test_input_fn():
	val_dataset = val_data

	iterator = val_dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	x = {'image': features}
	y = labels

	return x, y


feature_image = tf.feature_column.numeric_column('image', shape=[16384*3], dtype=tf.float32)
num_hidden_units = [512, 256, 128]
model = tf.estimator.DNNClassifier(feature_columns=[feature_image],
				hidden_units=num_hidden_units,
				activation_fn=tf.nn.relu,
				n_classes=num_classes,
				model_dir="./checkpoints")

iterations = (25000/batch_size) * num_epochs # for ease of use
model.train(input_fn=train_input_fn, steps=200)

# accuracy_score = model.evaluate(input_fn=test_input_fn)["accuracy"]
# print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
