import os
import torch
from gender_cnn import GenderCnn
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
latest_model = "cnn_epoch19.pkl"
state_dict = torch.load(os.path.join(model_dir, latest_model))

model = GenderCnn()
model.load_state_dict(state_dict)
classes = {0: "female", 1: "male"}
transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Makes a prediction.
image = load_single_image("test/min-yen.jpg", transformer)
output = model(image)
_, predict = torch.max(output.data, 1)
print("The prediction result is %s." % classes[int(predict)])
