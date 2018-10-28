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

RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
CHANNEL = 3

LOGGING_NAME = "sigmoid_tensor"

# Model function for CNN.
def cnn_model_fn(features, labels, mode):
	labels = tf.reshape(labels, [1, 1])

	# Defines the topology of the network here.
	input_layer = tf.reshape(features, [-1, RESIZE_HEIGHT, RESIZE_WIDTH, CHANNEL])
	print("The input layer size is %s" % input_layer.shape)

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv1 layer size is %s" % conv1.shape)

	conv2 = tf.layers.conv2d(
		inputs=conv1,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv2 layer size is %s" % conv2.shape)

	conv3 = tf.layers.conv2d(
		inputs=conv2,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print("The conv3 layer size is %s" % conv3.shape)

	pool1 = tf.layers.max_pooling2d(
		inputs=conv3,
		pool_size=[2, 2],
		strides=2)
	print("The pool1 layer size is %s" % pool1.shape)

	conv4 = tf.layers.conv2d(
		inputs=pool1,
		filters=128,
		kernel_size=[5, 5],
		strides=(2, 2),
		padding="valid",
		activation=tf.nn.relu)
	print("The conv4 layer size is %s" % conv4.shape)

	pool2 = tf.layers.max_pooling2d(
		inputs=conv4,
		pool_size=[3, 3],
		strides=2)
	print("The pool2 layer size is %s" % pool2.shape)

	conv5 = tf.layers.conv2d(
		inputs=pool2,
		filters=128,
		kernel_size=[3, 3],
		strides=(2, 2),
		padding="same",
		activation=tf.nn.relu)
	print("The conv5 layer size is %s" % conv5.shape)

	pool3 = tf.layers.max_pooling2d(
		inputs=conv5,
		pool_size=[3, 3],
		strides=2)
	print("The pool3 layer size is %s" % pool3.shape)

	conv6 = tf.layers.conv2d(
		inputs=pool3,
		filters=64,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	print("The conv6 layer size is %s" % conv6.shape)

	conv7 = tf.layers.conv2d(
		inputs=conv6,
		filters=32,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	print("The conv7 layer size is %s" % conv7.shape)

	pool4 = tf.layers.average_pooling2d(
		inputs=conv7,
		pool_size=[6, 6],
		strides=1)
	print("The pool4 layer size is %s" % pool4.shape)

	# Flats the tensor into a batch of vectors
	flat = tf.reshape(pool4, [-1, 1 * 1 * 32])
	print("The flatten pool4 size is %s" % flat.shape)

	# Dense (fully connected) Layer #1
	dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

	# Performs dropout operation here as a regularization technique.
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

	# Dense (fully connected) Layer #2 (Logits Layer)
	logits = tf.layers.dense(inputs=dropout, units=1)

	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  # "classes": tf.argmax(input=logits, axis=1),
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
	  "probabilities": tf.nn.sigmoid(logits, name=LOGGING_NAME)
	}

	# Returns the prediction estimator (for PREDICT mode)
	if mode == tf.estimator.ModeKeys.PREDICT:
	  	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculates Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

	# Configures the Training operation (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
													   predictions=tf.round(predictions["probabilities"]))}

	# Returns the evaluation estimator (for EVAL mode)
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

dataExtractDir = "WIDER_extract"
labelDir = "WIDER_label"
modelDir = "WIDER_model"
labelTrainFileName = "wider_attribute_trainval.json"
labelTestFileName = "wider_attribute_test.json"

# Loads the JSON files for labels.
def loadJson(path):
	with open(path) as f:
		return json.load(f)

def readAndResizeImageToTensor(imagePath):
    file = tf.read_file(imagePath)
    decoded = tf.image.decode_jpeg(file, channels=CHANNEL)
    return tf.image.resize_images(decoded, [RESIZE_HEIGHT, RESIZE_WIDTH])

def createDataset(prefix, images, size_limit=100):
	files = []
	labels = []

	# Counts the number of targets read in total.
	total = 0
	for image in images:
		# Skips those images which don't have the required prefix.
		if (not image["file_name"].startswith(prefix)):
			continue

		# Counts the number of targets in the current image.
		current_count = 0
		for target in image["targets"]:
			gender = int(target["attribute"][0])
			if (gender == 0):
				continue

			# Computes the file name of the current target.
			basename = os.path.splitext(os.path.basename(image["file_name"]))[0]
			targetPath = os.path.join(dataExtractDir, prefix, basename + "_" + str(current_count) + ".jpg")
			if (not os.path.isfile(targetPath)):
				print("WARNING: the target at %s does not exist." % targetPath)
				continue

			files.append(readAndResizeImageToTensor(targetPath))
			labels.append(int((gender + 1) / 2))
			current_count += 1
			total += 1

			if (total % 100 == 0):
				print("Had read %d targets and %d labels." % (total, total))

		if (total > size_limit):
				break

	print("Created a %s dataset with %d images and %d labels." % (prefix, len(files), len(labels)))
	return tf.data.Dataset.from_tensor_slices((files, labels))

def main(argv):
	currentFileName = os.path.join(labelDir, labelTrainFileName)
	data = loadJson(currentFileName)
	images = data['images']

	def train_input_fn():
		return createDataset("train/", images)

	def val_input_fn():
		return createDataset("val/", images)

	currentFileName2 = os.path.join(labelDir, labelTestFileName)
	data2 = loadJson(currentFileName2)
	images2 = data2['images']

	def test_input_fn():
		return createDataset("test/", images2)

	# Set up logging for predictions
	tensors_to_log = {"probabilities": LOGGING_NAME}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

	# Create the Estimator
	gender_classifier = tf.estimator.Estimator(
	    model_fn=cnn_model_fn,
	    model_dir=modelDir)

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

	# Evaluate the model and print results
	eval_results = gender_classifier.evaluate(input_fn=test_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()
