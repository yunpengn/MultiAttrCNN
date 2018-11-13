#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
#https://www.tensorflow.org/tutorials/estimators/cnn
#https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b
#https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers

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

import math
import numpy as np
import tensorflow as tf
import json
import os

sourceDir = "cropped_set"
annotationDir = "wider_attribute_annotation"
imageDir = "Image"
#trainJson = "gender_train_21255.json"
#testJson = "gender_test_27038.json"
#valJson = "gender_val_5107.json"
trainJson = "sleeve_train_21130.json"
testJson = "sleeve_test_27213.json"
valJson = "sleeve_val_5061.json"

#selected_number = 0
selected_number = 5

shuffle_buffer_size = 25000
batch_size = 32
prefetch_buffer_size = 32
cDrop = 0.1
image_size = 128
tf.logging.set_verbosity(tf.logging.INFO)

def parse_fn(filename, scenes, label):
	image = tf.image.decode_image(tf.read_file(filename), channels=3)
	image /= 255
	image -= 0.5
	
	image2 = tf.image.decode_image(tf.read_file(filename), channels=3)
	image2 /= 255
	image2 -= 0.5
	return (tf.concat([image, image2], 2), label)

def train_parse_fn(filename, scenes, label):
	image = tf.image.decode_image(tf.read_file(filename), channels=3)
	image = _augment_helper(image)
	image /= 255
	image -= 0.5
	
	image2 = tf.image.decode_image(tf.read_file(filename), channels=3)
	image2 = _augment_helper(image2)
	image2 /= 255
	image2 -= 0.5
	return (tf.concat([image, image2], 2), label)

def add_salt_pepper_noise(input):
	weights = np.ones([image_size, image_size, 3])
	bias = np.zeros([image_size, image_size, 3])
	for i in range(image_size):
		for j in range(image_size):
			if np.random.random() < 0.003:
				weights[i][j] = [0, 0, 0]
				given_bias = 0 if np.random.random() < 0.5 else 255
				bias[i][j] = [given_bias, given_bias, given_bias]
	T_weights = tf.convert_to_tensor(weights)
	T_bias = tf.convert_to_tensor(bias)
	result = tf.add(tf.multiply(input, weights), bias)
	return result
  
def _augment_helper(image):
	#flip
	result = tf.image.random_flip_left_right(image)
	#brightness
	#result = tf.image.random_brightness(result, max_delta= 0.1)
	#crop and translate
	#box = np.array([[np.random.random() * 0.2, np.random.random() * 0.2, 0.8 + np.random.random() * 0.2, 0.8 + np.random.random() * 0.2]], dtype=np.float32)
	#result = tf.reshape(tf.image.crop_and_resize(tf.reshape(result, [1, image_size, image_size, 3]), box, [0], [image_size, image_size]), [image_size, image_size, 3])
	#rotate
	#result = tf.contrib.image.rotate(result, np.random.random() * 15 * math.pi / 180, interpolation='BILINEAR')
	#salt and pepper
	result = add_salt_pepper_noise(result)	
	return result

def train_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + trainJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	scenes = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["scene"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][selected_number] + 1) / 2] for i in range(len(annotationDict))]
	dataset = tf.data.Dataset.from_tensor_slices((filenames, scenes, labels))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=train_parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def val_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + valJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	scenes = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["scene"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][selected_number] + 1) / 2] for i in range(len(annotationDict))]
	dataset = tf.data.Dataset.from_tensor_slices((filenames, scenes, labels))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def test_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + testJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	scenes = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["scene"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][selected_number] + 1) / 2] for i in range(len(annotationDict))]
	dataset = tf.data.Dataset.from_tensor_slices((filenames, scenes, labels))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	l2_reg = tf.contrib.layers.l2_regularizer(scale=0.00)

	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	features = tf.reshape(features, [-1, image_size, image_size, 6])
	input_layer, context_layer = tf.split(features, [3, 3], 3)

	att1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=8,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	att2 = tf.layers.conv2d(
		inputs=att1,
		filters=8,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	att3 = tf.layers.conv2d(
		inputs=att2,
		filters=1,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.sigmoid)

	filtered = tf.multiply(input_layer, att3)
	conv1a = tf.layers.conv2d(
		inputs=filtered,
		filters=8,
		kernel_size=[5, 5],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1a, pool_size=[3, 3], strides=2)
	norm1 = tf.nn.lrn(pool1)

	conv2a = tf.layers.conv2d(
		inputs=norm1,
		filters=8,
		kernel_size=[5, 5],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2a, pool_size=[3, 3], strides=2)
	norm2 = tf.nn.lrn(pool2)

	conv3a = tf.layers.conv2d(
		inputs=norm2,
		filters=8,
		kernel_size=[5, 5],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(inputs=conv3a, pool_size=[3, 3], strides=2)
	norm3 = tf.nn.lrn(pool3)

	conv4a = tf.layers.conv2d(
		inputs=norm3,
		filters=16,
		kernel_size=[5, 5],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	pool4 = tf.layers.max_pooling2d(inputs=conv4a, pool_size=[3, 3], strides=2)
	norm4 = tf.nn.lrn(pool4)

	context_conv1a = tf.layers.conv2d(
		inputs=context_layer,
		filters=8,
		kernel_size=[3, 3],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	context_pool1 = tf.layers.max_pooling2d(inputs=context_conv1a, pool_size=[3, 3], strides=2)
	context_norm1 = tf.nn.lrn(context_pool1)

	context_conv2a = tf.layers.conv2d(
		inputs=tf.layers.dropout(context_norm1, rate=cDrop, training=mode == tf.estimator.ModeKeys.TRAIN),
		filters=8,
		kernel_size=[3, 3],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	context_pool2 = tf.layers.max_pooling2d(inputs=context_conv2a, pool_size=[3, 3], strides=2)
	context_norm2 = tf.nn.lrn(context_pool2)

	context_conv3a = tf.layers.conv2d(
		inputs=tf.layers.dropout(context_norm2, rate=cDrop, training=mode == tf.estimator.ModeKeys.TRAIN),
		filters=8,
		kernel_size=[3, 3],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	context_pool3 = tf.layers.max_pooling2d(inputs=context_conv3a, pool_size=[3, 3], strides=2)
	context_norm3 = tf.nn.lrn(context_pool3)

	context_conv4a = tf.layers.conv2d(
		inputs=tf.layers.dropout(context_norm3, rate=cDrop, training=mode == tf.estimator.ModeKeys.TRAIN),
		filters=16,
		kernel_size=[3, 3],
		kernel_regularizer=l2_reg,
		padding="same",
		activation=tf.nn.relu)
	context_pool4 = tf.layers.max_pooling2d(inputs=context_conv4a, pool_size=[3, 3], strides=2)
	context_norm4 = tf.nn.lrn(context_pool4)

	flat_layer = tf.reshape(tf.concat([norm4, context_norm4], 3), [-1, 7 * 7 * 32])
#	flat_layer = tf.reshape(norm4, [-1, 7 * 7 * 16])


	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 8 * 8 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=flat_layer, units=1024, activation=tf.nn.relu)

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
	#l2_loss = tf.losses.get_regularization_loss()
	#loss += l2_loss

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
		model_fn=cnn_model_fn, model_dir="model/longsleeves_model")

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
		throttle_secs=120,
		start_delay_secs=120
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
