from age_cnn import AgeCnn
from gender_cnn import GenderCnn
import flask
import io
import os
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

# Instantiates a Flask app.
app = flask.Flask(__name__)
# Determines the folder in which all models are stored.
model_dir = "model"

# Defines the class-to-label mapping.
classes = {"gender": ["female", "male"],
           "age": ['0', '10', '15', '20', '25', '30', '35', '40', '45', '5', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']}
model = {"gender": GenderCnn(), "age": AgeCnn()}
transformer = {"gender": transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
               "age": transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])}

POST_PARAM = "face_image"


# Loads all models into memory.
def load_model():
    global model

    for key in model:
        state_dict = torch.load(os.path.join(model_dir, key + ".pkl"))
        model[key].load_state_dict(state_dict)
        model[key].eval()


# Loads a single image from HTTP POST request.
def load_image(attribute, image):
    transformed = transformer[attribute](image).float()
    variable = Variable(transformed)
    return variable.unsqueeze(0)


# Given an image, predicts a certain human attribute.
def predict_attribute(attribute, image):
    output = model[attribute](image)
    _, prediction = torch.max(output.data, 1)
    return classes[attribute][int(prediction)]


@app.route("/predict", methods=["POST"])
def predict():
    # Initializes the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensures an image was properly uploaded to our endpoint.
    if flask.request.method == "POST" and flask.request.files.get(POST_PARAM):
        # Reads the image in PIL format
        parameter = flask.request.files[POST_PARAM].read()
        image = Image.open(io.BytesIO(parameter))

        # Calculates multiple human attributes.
        data["gender"] = predict_attribute("gender", load_image("gender", image))
        data["age"] = predict_attribute("age", load_image("age", image))
        data["success"] = True

    # Return the data dictionary as a JSON response.
    print("Sent prediction result %s" % data)
    return flask.jsonify(data)


if __name__ == '__main__':
    load_model()
    app.run()
