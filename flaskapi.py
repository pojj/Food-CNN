import torch, json, io
from PIL import Image
from flask import Flask, jsonify, request
from format_image import FormatImage
from model import NeuralNetwork


def format_bytes(image):
    if type(image) is bytes:
        image = Image.open(io.BytesIO(image))
        data_in = img_format(image)
        data_in = data_in.unsqueeze(0)
        return data_in


def get_prediction(data):
    with torch.no_grad():
        pred = model(data).softmax(1)
        confidence, guess = pred.max(1)
        return confidence.item(), LABEL_DICT[str(guess.item())]


app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        img_bytes = file.read()
        data_in = format_bytes(img_bytes)
        confidence, label_name = get_prediction(data_in)
        return jsonify({"class_name": label_name, "confidence": confidence})


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response


with open("data\\labeldict.json", "r") as l:
    LABEL_DICT = json.load(l)

IMAGE_SIZE = 256

img_format = FormatImage(IMAGE_SIZE)

model = NeuralNetwork()
for param in model.parameters():
    param.requires_grad_(False)

model.load_state_dict(torch.load("models\\all50-256\\all7(79.7%).pth"))
model.eval()


if __name__ == "__main__":
    app.run()
