# CS3244 Project

## Notes regarding the WIDER dataset

WIDER Attribute is a large-scale human attribute dataset. It contains 13789 images belonging to 30 scene categories, and 57524 human bounding boxes each annotated with 14 binary attributes.

The annotations are stored in JSON file format. The data structure is:

```
{
	"images"		: [image],					# A list of "image" (see below)
	"attribute_id_map"	: {str : str},			# A dict, mapping attribute_id to attribute_name
	"scene_id_map"		: {str : str}			# A dict, mapping scene_id to scene_name
}

image {
	"targets"		: [target],			# A list of "target" (see below)
	"file_name"		: str,				# Image file name
	"scene_id"		: int				# Scene id
}

target {
	"attribute"		: [int]				# A list of int, the i-th element corresponds to the i-th attribute, and the value could be 1(possitive), -1(negative) or 0 (unspecified)
	"bbox"			: [x, y, width, height] 	# Human bounding box
}
```

For more, see http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html

## Licence

[GNU General Public Licence 3.0](LICENSE)