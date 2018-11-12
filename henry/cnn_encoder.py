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

import numpy as np
import tensorflow as tf
import json
import os
import cv2

sourceDir = "cropped_set"
annotationDir = "wider_attribute_annotation"
imageDir = "Image"
trainJson = "train_21255.json"
testJson = "test_27038.json"
valJson = "val_5107.json"

shuffle_buffer_size = 25000
batch_size = 256
prefetch_buffer_size = 256
tf.logging.set_verbosity(tf.logging.INFO)

def parse_fn(filename, label):
	image = tf.image.decode_image(tf.read_file(filename), channels=1)
	tf.Tensor.set_shape(image, (128, 128, 1))
	image = tf.image.resize_images(image, (32, 32))
 	#image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear
	image /= 255
	image -= 0.5
	return (image, label)

def train_input_fn():
	with open(sourceDir + "/" + annotationDir + "/" + trainJson, 'r') as f:
		annotationDict = json.load(f)
	filenames = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	labels = [[(annotationDict[str(i)]["attribute"][0] + 1) / 2] for i in range(len(annotationDict))]
	with open(sourceDir + "/" + annotationDir + "/" + testJson, 'r') as f:
		annotationDict = json.load(f)
	filenames2 = [sourceDir + "/" + imageDir + "/" + annotationDict[str(i)]["filename"] for i in range(len(annotationDict))]
	labels2 = [[(annotationDict[str(i)]["attribute"][0] + 1) / 2] for i in range(len(annotationDict))]
	filenames += filenames2
	labels += labels2
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
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
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	return dataset

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""

	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	original = tf.reshape(features, [-1, 32, 32, 1])
	
	# encoder
	# 32 x 32 x 1   ->  16 x 16 x 32
	# 16 x 16 x 32  ->  8 x 8 x 16
	# 8 x 8 x 16    ->  2 x 2 x 8
	net = tf.layers.conv2d(original, 32, [5, 5], strides=[2, 2], padding='SAME')
	net = tf.layers.conv2d(net, 16, [5, 5], strides=[2, 2], padding='SAME')
	net = tf.layers.conv2d(net, 8, [5, 5], strides=[4, 4], padding='SAME')
	# decoder
	# 2 x 2 x 8    ->  8 x 8 x 16
	# 8 x 8 x 16   ->  16 x 16 x 32
	# 16 x 16 x 32  ->  32 x 32 x 1
	net = tf.layers.conv2d_transpose(net, 16, [5, 5], strides=[4, 4], padding='SAME')
	net = tf.layers.conv2d_transpose(net, 32, [5, 5], strides=[2, 2], padding='SAME')
	net = tf.layers.conv2d_transpose(net, 1, [5, 5], strides=[2, 2], padding='SAME', activation=tf.nn.tanh)
	
	result = net

	if mode == tf.estimator.ModeKeys.PREDICT:
		# Add predict metrics (for predict mode)
		predictions = {"original" : original, "result" : result}
		return tf.estimator.EstimatorSpec(
			mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.mean_squared_error(original, net)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add predict metrics (for predict mode)
	predictions = {"original" : original, "result" : result}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, predictions=predictions)


def main(unused_argv):
	# Create the Estimator
	gender_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="model/cnn_encoder")

	# Set up logging for predictions
	# Log the values in the "Sigmoid" tensor with label "probabilities"
	tensors_to_log = {}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	#  gender_classifier.train(
	#      input_fn=train_input_fn,
	#      hooks=[logging_hook])
	train_spec = tf.estimator.TrainSpec(
		input_fn=train_input_fn,
		hooks=[logging_hook],
		max_steps=1500
	)
	val_spec = tf.estimator.EvalSpec(
		input_fn=val_input_fn,
		hooks=[logging_hook],
		throttle_secs=90,
		start_delay_secs=90
	)
	tf.estimator.train_and_evaluate(
		gender_classifier,
		train_spec,
		val_spec
	)

	# predict the model and print results
	predict_results = gender_classifier.predict(input_fn=val_input_fn)
	total = 0
	for predict_result in predict_results:
		image = predict_result["original"]
		image += 0.5
		image *= 255
		cv2.imwrite("given" + str(total) + ".jpg", np.uint8(image))
		image = predict_result["result"]
		image += 0.5
		image *= 255
		cv2.imwrite("test" + str(total) + ".jpg", np.uint8(image))

		total += 1
		if total > 10:
			break


if __name__ == "__main__":
  tf.app.run()
