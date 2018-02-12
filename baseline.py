import tensorflow as tf
import numpy as np
from input_pipe import tr_data, val_data, training_init_op, validation_init_op, next_element, train_imgs, train_labels, val_imgs, val_labels

def input_fn_train(imgs=train_imgs, labels=train_labels):
	return imgs, labels

def input_fn_val(imgs=val_imgs, labels=val_labels):
	return imgs, labels

# Build BaselineClassifier
classifier = tf.estimator.BaselineClassifier(n_classes=2)

# Fit model.
classifier.train(input_fn=input_fn_train)

# Evaluate cross entropy between the test and train labels.
loss = classifier.evaluate(input_fn=input_fn_val)["loss"]

# predict outputs the probability distribution of the classes as seen in
# training.
predictions = classifier.predict(new_samples)

