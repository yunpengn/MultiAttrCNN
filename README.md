# CS3244 Project

## Notes regarding the WIDER dataset

WIDER Attribute is a large-scale human attribute dataset. It contains 13789 images belonging to 30 scene categories, and 57524 human bounding boxes each annotated with 14 binary attributes.

The annotations are stored in JSON file format. The data structure is:

```
{
	"images"            : [image],          # A list of "image" (see below)
	"attribute_id_map"  : {str : str},      # A dict, mapping attribute_id to attribute_name
	"scene_id_map"		: {str : str}       # A dict, mapping scene_id to scene_name
}

image {
	"targets"           : [target],			# A list of "target" (see below)
	"file_name"         : str,				# Image file name
	"scene_id"          : int				# Scene id
}

target {
	"attribute"         : [int]             # A list of int, the i-th element corresponds to the i-th attribute, and the value could be 1(possitive), -1(negative) or 0(unspecified)
	"bbox"              : [x, y, width, height] # Human bounding box
}
```

For more, see http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html

## Using [Docker](https://www.docker.com) to install Tensorflow

- Sometimes, it can be quite troublesome to install tensorflow on your machine. If so, you can use Docker to release the pain.
- Pull the Docker image by `docker pull tensorflow/tensorflow:latest-py3`
- Let's first create a container by `docker run -it --name tensorflow -p 0.0.0.0:7007:6006 -v $PWD:/CS3244 tensorflow/tensorflow:latest-py3 bash`.
	- `it` is always used for interactive command, like a shell.
	- `rm` will delete the container automatically after exit.
	- `v` and `w` specifies how the directory sharing is done.
	- `p` will do port forwarding (for tensorboard later). Use `tensorboard --logdir ./ --host 0.0.0.0` to run tensorboard.
- From then on, you can start the container by `docker start tensorflow`. Then, you can login by `docker exec -it tensorflow bash`. To stop the container, run `docker stop tensorflow`.

## Licence

[GNU General Public Licence 3.0](LICENSE)