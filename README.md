# CS3244 Project

This project is participating in NUS SoC 13th STePS. For more information, see [here](http://isteps.comp.nus.edu.sg/event/13th-steps/module/CS3244/project/2).

Information about the repository structure:
- [Project Report](report.pdf)
- [Gender Classifier for the LFW dataset](torch/)
    - Implemented using [PyTorch](https://pytorch.org) library.
    - An older model can be found at [here](yunpeng_old/)
- [Long/short-sleeves Classifier for the WIDER dataset](henry/)
    - Implemented using [TensorFlow](https://www.tensorflow.org) library.

## Notes regarding the WIDER dataset

See [here](WIDER.md).

## Notes regarding the LFW dataset

See [here](LFW.md).

## Using [Docker](https://www.docker.com) to install Tensorflow

- Sometimes, it can be quite troublesome to install tensorflow on your machine. If so, you can use Docker to release the pain.
- Pull the Docker image by `docker pull tensorflow/tensorflow:latest-py3`
- Let's first create a container by `docker run -it --name tensorflow -p 0.0.0.0:7007:6006 -v $PWD:/CS3244 tensorflow/tensorflow:latest-py3 bash`.
	- `it` is always used for interactive command, like a shell.
	- `rm` will delete the container automatically after exit.
	- `v` and `w` specifies how the directory sharing is done.
	- `p` will do port forwarding (for tensorboard later). Use `tensorboard --logdir ./ --host 0.0.0.0` to run tensorboard. Then you can see the board at [http://localhost:7007](http://localhost:7007).
- From then on, you can start the container by `docker start tensorflow`. Then, you can login by `docker exec -it tensorflow bash`. To stop the container, run `docker stop tensorflow`.

## Resources

- http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- http://vis-www.cs.umass.edu/lfw/

## Licence

[GNU General Public Licence 3.0](LICENSE)
