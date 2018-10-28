import os
import torch
from convolution_net import ConvolutionNet
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms


def load_single_image(image_path, transform):
    loaded = Image.open(image_path)
    transformed = transform(loaded).float()
    variable = Variable(transformed)
    return variable.unsqueeze(0)


if __name__ != '__main__':
    print("This module must be run as the main module.")
    exit(1)

# Loads the model and related variables.
model_dir = "LFW_model_torch"
latest_model = "cnn_epoch1.pkl"
state_dict = torch.load(os.path.join(model_dir, latest_model))

model = ConvolutionNet()
model.load_state_dict(state_dict)
transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Makes a prediction.
image = load_single_image("test.jpg", transformer)
predict = model(image)
result = torch.max(predict.data, 1)
