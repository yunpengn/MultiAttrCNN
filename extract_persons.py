import json
import numpy as np
import os
import pandas as panda
import sys
import tensorflow as tf
import traceback

from PIL import Image

dataDir = "data"
dataExtractDir = "extract"
labelDir = "label"
labelTrainFileName = "wider_attribute_trainval.json"
labelTestFileName = "wider_attribute_test.json"

# Begin the session
session = tf.Session()

# Sets the logging level for TensorFlow library.
tf.logging.set_verbosity(tf.logging.INFO)

imageCount = [0]

# Loads the JSON files for labels.
def loadJson(path):
    with open(path) as f:
        return json.load(f)

def loadJsonUrl(url):
    with request.urlopen(url) as link:
        return json.load(link.read().decode())

def readImageToTensor(imagePath):
    file = tf.read_file(imagePath)
    return tf.image.decode_jpeg(file, channels=3)

def correctBboxValues(imageTensor, bbox):
    offsetHeight = int(bbox[1])
    offsetWidth  = int(bbox[0])
    targetHeight = int(bbox[3])
    targetWidth  = int(bbox[2])
    return [offsetHeight, offsetWidth, targetHeight, targetWidth]

def extractPersonFromImage(imageData):
    imagePath = os.path.join(dataDir, imageData["file_name"])
    imageTensor = readImageToTensor(imagePath)
    session.run(imageTensor)

    i = 0
    for target in imageData["targets"]:
        attributes = target["attribute"]
        bbox = correctBboxValues(imageTensor, target["bbox"])

        try:
            croppedTensor = tf.image.crop_to_bounding_box(imageTensor, bbox[0], bbox[1], bbox[2], bbox[3])
            encodedTensor = tf.image.encode_jpeg(croppedTensor)
            outputPath = os.path.join(dataExtractDir, imageData["file_name"], str(i) + ".jpg")
            newImage = tf.write_file(tf.constant(outputPath), encodedTensor)
            session.run(newImage)
        except:
            print("Unable to extract person around bbox %s from the image at %s." % (bbox, imageData["file_name"]))
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)

        i = i + 1

    print("Extract %d person(s) from the image at %s." % (i + 1, imageData["file_name"]))
    imageCount[0] += (i + 1)
    print("Cropped %d images up to now." % (imageCount[0]))

currentFileName = os.path.join(labelDir, labelTrainFileName)
data = loadJson(currentFileName)
images = data['images']
attributeIdMap = data['attribute_id_map']
sceneIdMap = data['scene_id_map']

for image in images:
    if (image["file_name"].startswith("val/0--Parade/")):
        extractPersonFromImage(image)

# Close the session after use.
session.close()
