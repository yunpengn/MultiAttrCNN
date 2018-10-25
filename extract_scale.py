import os
import json
import pickle
import numpy as np
import re
import cv2

sourceDir = "raw_set"
targetDir = "cropped_set"
imageDir = "Image"
annotationDir = "wider_attribute_annotation"
categories = 30

targetSize = 128

def createDirs():
	for topDir in os.listdir(sourceDir + "/" + imageDir):
		if not os.path.exists(targetDir + "/" + imageDir + "/" + topDir):
			os.mkdir(targetDir + "/" + imageDir + "/" + topDir)
		for dir in os.listdir(sourceDir + "/" + imageDir + "/" + topDir):
			if not os.path.exists(targetDir + "/" + imageDir + "/" + topDir + "/" + dir):
				os.mkdir(targetDir + "/" + imageDir + "/" + topDir + "/" + dir)

def cropImage(image, target):
	#target
	#["attribute", "bbox"]
	bbox = target["bbox"]
	#x, y cannot be outside
	bbox[0] = min(max(0, bbox[0]), image.shape[1] - 1)
	bbox[1] = min(max(0, bbox[1]), image.shape[0] - 1)
	#fix negative width and height
	if (bbox[2] < 0):
		bbox[0] += bbox[2]
		bbox[2] = -bbox[2]
	if (bbox[3] < 0):
		bbox[1] += bbox[3]
		bbox[3] = -bbox[3]
	bbox = [int(val) for val in bbox]
	#Fix exceeding box boundaries
	bbox[2] = min(bbox[2], image.shape[1] - bbox[0])
	bbox[3] = min(bbox[3], image.shape[0] - bbox[1])
	
	#Crop the image and resize
	croppedImage = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
	height = croppedImage.shape[0]
	width = croppedImage.shape[1]
	resized_image = cv2.resize(croppedImage, (targetSize, targetSize))
	return resized_image

def parseAnnotationJson(filename, dataGroups):
	sourceJson = sourceDir + "/" + annotationDir + "/" + filename
	targetJson = targetDir + "/" + annotationDir + "/"
	data = {}
	for dataGroup in dataGroups:
		data[dataGroup] = {}
		for id in range(categories):
			data[dataGroup][str(id)] = []
	with open(sourceJson, 'r') as f:
		annotationDict = json.load(f)

	#annotationDict
	#["images", "attribute_id_map", "scene_id_map"]
	
	count = [0] * categories
	for compositeImage in annotationDict["images"]:
		#compositeImage
		#["scene_id", "file_name", "targets"]
		sceneid = compositeImage["scene_id"]
		filename = compositeImage["file_name"]
		targets = compositeImage["targets"]
		if not os.path.exists(sourceDir + "/" + imageDir + "/" + filename):
			continue
		image = cv2.imread(sourceDir + "/" + imageDir + "/" + filename, 1)
		for target in targets:
			#target
			#["attribute", "bbox"]
			newFilename = re.match("^.*/", filename).group(0) + str(count[sceneid]) + ".jpg"
			#if not os.path.exists(targetDir + "/" + imageDir + "/" + newFilename):
			if True:
				newImage = cropImage(image, target)
				if newImage is None:
					continue
				cv2.imwrite(targetDir + "/" + imageDir + "/" + newFilename, newImage)
			for dataGroup in dataGroups:
				if dataGroup in filename:
					data[dataGroup][str(sceneid)].append({"filename" : newFilename, "attribute" : target["attribute"]})
					break;
			count[sceneid] += 1
	
	for dataGroup in dataGroups:
		with open(targetJson + dataGroup + ".json", "w") as f:
			json.dump(data[dataGroup], f)
	
def create_storage(data, name, storage_size):
    i = 0
    while i * storage_size <= len(data):
        with open(name+ '_' + str(i) +'.pickle', 'wb') as handle:
            content = data[(i * storage_size):((i+1) * storage_size)]
            pickle.dump(content, handle)
            print('Saved',name,'part #' + str(i), 'with', len(content),'entries.')
        i += 1

createDirs()
parseAnnotationJson("wider_attribute_test.json", ["test"])
parseAnnotationJson("wider_attribute_trainval.json", ["train", "val"])

