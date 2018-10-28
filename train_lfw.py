# Imports the relevant libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import os.path
import re
import sys
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib

# Sets the logging level for TensorFlow library.
tf.logging.set_verbosity(tf.logging.INFO)

RESIZE_HEIGHT = 32
RESIZE_WIDTH = 32
CHANNEL = 3
BATCH = -1
extract_dir = "LFW_extract"
label_dir = "LFW_label"
label_male_names = "male_names.txt"
label_female_names = "female_names.txt"
model_dir = "LFW_model"
LOGGING_NAME = "sigmoid_tensor"

# Model function for CNN.
def cnn_model_fn(features, labels, mode):
	labels = tf.reshape(labels, [1, 1])

	# Defines the topology of the network here.
	input_layer = tf.reshape(features, [BATCH, RESIZE_HEIGHT, RESIZE_WIDTH, CHANNEL])
	print("The input layer size is %s" % input_layer.shape)

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv1 layer size is %s" % conv1.shape)

	conv2 = tf.layers.conv2d(
		inputs=conv1,
		filters=32,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv2 layer size is %s" % conv2.shape)

	pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	print("The pool1 layer size is %s" % conv2.shape)

	conv3 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv3 layer size is %s" % conv3.shape)

	conv4 = tf.layers.conv2d(
		inputs=conv3,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv4 layer size is %s" % conv4.shape)

	pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
	print("The pool2 layer size is %s" % pool2.shape)

	flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
	print("The flat layer size is %s" % flat.shape)

	dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout, units=1)

	predictions = {
		# Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
		"probabilities": tf.nn.sigmoid(logits, name=LOGGING_NAME)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=tf.round(predictions["probabilities"]))}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def readAndResizeImageToTensor(imagePath, do_resize=True):
    file = tf.read_file(imagePath)
    decoded = tf.image.decode_jpeg(file, channels=CHANNEL)

    if do_resize:
    	return tf.image.resize_images(decoded, [RESIZE_HEIGHT, RESIZE_WIDTH])
    else:
    	return decoded

def list_files(folder):
	return [os.path.join(folder, f) for f in os.listdir(folder)
									if os.path.isfile(os.path.join(folder, f)) and f.endswith(".jpg")]

# Label female => 0, male => 1
def createDataset(male_folder, female_folder):
	files = [readAndResizeImageToTensor(f) for f in list_files(male_folder)]
	labels = [1] * len(files)

	female = [readAndResizeImageToTensor(f) for f in list_files(female_folder)]
	files += female
	labels += [0] * len(female)

	print("Going to create a dataset with %d images and %d labels." % (len(files), len(labels)))
	return tf.data.Dataset.from_tensor_slices((files, labels))

def main(argv):
	def train_input_fn():
		return createDataset(os.path.join(extract_dir, "train_male/"), os.path.join(extract_dir, "train_female/"))

	def val_input_fn():
		return createDataset(os.path.join(extract_dir, "val_male/"), os.path.join(extract_dir, "val_female/"))

	# Set up logging for predictions
	tensors_to_log = {"probabilities": LOGGING_NAME}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

	# Create the Estimator
	gender_classifier = tf.estimator.Estimator(
	    model_fn=cnn_model_fn,
	    model_dir=model_dir)

	train_spec = tf.estimator.TrainSpec(
		input_fn=train_input_fn,
		hooks=[logging_hook],
		max_steps=2000
	)

	val_spec = tf.estimator.EvalSpec(
		input_fn=val_input_fn,
		hooks=[logging_hook],
		throttle_secs=15,
		start_delay_secs=15
	)

	# Train the model (can increase the number of steps to improve accuracy)
	tf.estimator.train_and_evaluate(gender_classifier, train_spec, val_spec)

if __name__ == "__main__":
	tf.app.run()
