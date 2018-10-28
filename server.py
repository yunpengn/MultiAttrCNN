from convolution_net import ConvolutionNet
import flask
import io
import os
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

app = flask.Flask(__name__)
classes = {0: "female", 1: "male"}
transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
model = ConvolutionNet()

POST_PARAM = "face_image"


def load_model():
    global model

    model_dir = "LFW_model_torch"
    latest_model = "cnn_epoch19.pkl"
    state_dict = torch.load(os.path.join(model_dir, latest_model))

    model.load_state_dict(state_dict)
    model.eval()


def load_image(image):
    loaded = Image.open(io.BytesIO(image))
    transformed = transformer(loaded).float()
    variable = Variable(transformed)
    return variable.unsqueeze(0)


def predict_gender(image):
    output = model(image)
    _, prediction = torch.max(output.data, 1)
    return classes[int(prediction)]


@app.route("/gender_predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST" and flask.request.files.get(POST_PARAM):
        # Read the image in PIL format
        parameter = flask.request.files[POST_PARAM].read()
        image = load_image(parameter)
        result = predict_gender(image)

        # Returns the result.
        data["gender"] = result
        data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    load_model()
    app.run()
