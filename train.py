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

RESIZE_HEIGHT = 28
RESIZE_WIDTH = 28

# Model function for CNN.
def cnn_model_fn(features, labels, mode):
	labels = tf.reshape(labels, [1, 1])

	# Input Layer
	input_layer = tf.reshape(features, [-1, RESIZE_HEIGHT, RESIZE_WIDTH, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=[2, 2],
		strides=2)

	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=[2, 2],
		strides=2)

	# Dense Layer #1
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	dense      = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout    = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Dense Layer #2 (Logits Layer)
	logits = tf.layers.dense(inputs=dropout, units=1)

	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  # "classes": tf.argmax(input=logits, axis=1),
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
	  "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
	  }

	if mode == tf.estimator.ModeKeys.PREDICT:
	  	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=tf.round(predictions["probabilities"]))}

	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

dataExtractDir = "extract"
labelDir = "label"
labelTrainFileName = "wider_attribute_trainval.json"
labelTestFileName = "wider_attribute_test.json"

# Loads the JSON files for labels.
def loadJson(path):
	with open(path) as f:
		return json.load(f)

def readAndResizeImageToTensor(imagePath):
    file = tf.read_file(imagePath)
    decoded = tf.image.decode_jpeg(file, channels=1)
    return tf.image.resize_images(decoded, [RESIZE_HEIGHT, RESIZE_WIDTH])

def createDataset(prefix, images):
	files = []
	labels = []

	for image in images:
		if (not image["file_name"].startswith(prefix)):
			continue

		i = 0
		for target in image["targets"]:
			gender = int(target["attribute"][0])
			if (gender == 0):
				continue

			basename = os.path.splitext(os.path.basename(image["file_name"]))[0]
			targetPath = os.path.join(dataExtractDir, prefix, basename + "_" + str(i) + ".jpg")

			files.append(readAndResizeImageToTensor(targetPath))
			labels.append(int((gender + 1) / 2))
			i += 1

	print("Created a train dataset with %d images and %d labels." % (len(files), len(labels)))
	return tf.data.Dataset.from_tensor_slices((files, labels))

def main(argv):
	currentFileName = os.path.join(labelDir, labelTrainFileName)
	data = loadJson(currentFileName)
	images = data['images']
	attributeIdMap = data['attribute_id_map']
	sceneIdMap = data['scene_id_map']

	def train_input_fn():
		return createDataset("val/0--Parade/", images)

	def val_input_fn():
		return createDataset("val/1--Handshaking/", images)

	def test_input_fn():
		return createDataset("val/3--Riot/", images)

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "sigmoid_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=200)

	# Create the Estimator
	gender_classifier = tf.estimator.Estimator(
	    model_fn=cnn_model_fn,
	    model_dir="model")

	# Train the model (can increase the number of steps to improve accuracy)
	gender_classifier.train(
	    input_fn=train_input_fn,
	    steps=500,
	    hooks=[logging_hook])

	gender_classifier.evaluate(
		input_fn=val_input_fn,
	    hooks=[logging_hook])

	# Evaluate the model and print results
	eval_results = gender_classifier.evaluate(input_fn=test_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()
