from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "fruit_freshness_model.h5"

# Load model
model = load_model(MODEL_PATH)

# Change these according to your trained model
classes = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image selected"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    # Open image
    image = Image.open(file).convert("RGB")

    # Your model uses 150 x 150 images
    image = image.resize((150, 150))

    # Convert image to numpy array
    image_array = np.array(image) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction)) * 100

    result = classes[predicted_class]

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
