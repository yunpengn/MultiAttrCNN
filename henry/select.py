import os
import json
import numpy as np

sourceDir = "cropped_set/wider_attribute_annotation"

def selectAnnotationJson(sourceJson):
	with open(sourceDir + "/" + sourceJson, 'r') as f:
		annotationDict = json.load(f)
	#annotationDict
	#["0" ..., "29"]
	
	data = {}
	count = 0
	for key in annotationDict.keys():
		images = np.array(annotationDict[key])
		np.random.shuffle(images)
		for index in range(int(len(images) / 4)):
			#images[index]
			#["filename", "attribute"]
			if not os.path.exists("cropped_set/Image/" + images[index]["filename"]) \
				or images[index]["attribute"][0] == 0:
				continue
			data[str(count)] = images[index]
			count += 1
	
	targetJson = sourceJson[:-5] + "_" + str(count) + ".json"
	with open(sourceDir + "/" + targetJson, "w") as f:
		json.dump(data, f)
	
selectAnnotationJson("train.json")
selectAnnotationJson("test.json")
selectAnnotationJson("val.json")
