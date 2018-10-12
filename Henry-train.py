#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
#https://www.tensorflow.org/tutorials/estimators/cnn
#https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b

#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
import os

sourceDir = "cropped_set"
annotationDir = "wider_attribute_annotation"
imageDir = "Image"
#trainJson = "train_5730.json"
#testJson = "test_7282.json"
#valJson = "val_1335.json"
trainJson = "train_268.json"
testJson = "test_363.json"
valJson = "val_74.json"

shuffle_buffer_size = 1000
batch_size = 64
prefetch_buffer_size = 64
epochs = 100
tf.logging.set_verbosity(tf.logging.INFO)

def parse_fn(filename, label):
	image = tf.image.decode_image(tf.read_file(filename))
 	#image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear
	image /= 255
	image -= 0.5
	return (image, label)

def train_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + trainJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][0] + 1) / 2] for i in range(len(annotationDict))]
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
		buffer_size=shuffle_buffer_size, count=epochs))
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def val_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + valJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][0] + 1) / 2] for i in range(len(annotationDict))]
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
		buffer_size=shuffle_buffer_size, count=1))
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def test_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + testJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][0] + 1) / 2] for i in range(len(annotationDict))]
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
		buffer_size=shuffle_buffer_size, count=1))
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(features, [-1, 128, 128, 3])

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 128, 128, 3]
	# Output Tensor Shape: [batch_size, 128, 128, 32]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 128, 128, 32]
	# Output Tensor Shape: [batch_size, 64, 64, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 64, 64, 32]
	# Output Tensor Shape: [batch_size, 64, 64, 32]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 64, 64, 32]
	# Output Tensor Shape: [batch_size, 32, 32, 32]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Convolutional Layer #3
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 32, 32, 32]
	# Output Tensor Shape: [batch_size, 32, 32, 32]
	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #3
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 32, 32, 32]
	# Output Tensor Shape: [batch_size, 16, 16, 32]
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

	# Convolutional Layer #4
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 16, 16, 32]
	# Output Tensor Shape: [batch_size, 16, 16, 64]
	conv4 = tf.layers.conv2d(
		inputs=pool3,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #4
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 16, 16, 64]
	# Output Tensor Shape: [batch_size, 8, 8, 64]
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 8, 8, 64]
	# Output Tensor Shape: [batch_size, 8 * 8 * 64]
	pool4_flat = tf.reshape(pool4, [-1, 8 * 8 * 64])

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 8 * 8 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
	  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 1]
	logits = tf.layers.dense(inputs=dropout, units=1)

	predictions = {
		# Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
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


def main(unused_argv):
	# Create the Estimator
	gender_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="model/gender_model")

	# Set up logging for predictions
	# Log the values in the "Sigmoid" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "sigmoid_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	#  gender_classifier.train(
	#      input_fn=train_input_fn,
	#      hooks=[logging_hook])
	train_spec = tf.estimator.TrainSpec(
		input_fn=train_input_fn,
		hooks=[logging_hook],
		max_steps=100000
	)
	val_spec = tf.estimator.EvalSpec(
		input_fn=val_input_fn,
		hooks=[logging_hook],
		throttle_secs=15,
		start_delay_secs=15
	)
	tf.estimator.train_and_evaluate(
		gender_classifier,
		train_spec,
		val_spec
	)

	# Evaluate the model and print results
	eval_results = gender_classifier.evaluate(input_fn=test_input_fn)
	print(eval_results)


if __name__ == "__main__":
  tf.app.run()
