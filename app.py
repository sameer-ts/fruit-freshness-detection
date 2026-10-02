from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# TensorFlow Lite model
MODEL_PATH = "fruit_freshness_model.tflite"

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# IMPORTANT:
# These must match the classes used when your model was trained.
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

    try:
        # Open uploaded image
        image = Image.open(file).convert("RGB")

        # Your original model uses 150 x 150 images
        image = image.resize((150, 150))

        # Convert image to NumPy
        image_array = np.array(image, dtype=np.float32)

        # Normalize
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Send image to TensorFlow Lite
        interpreter.set_tensor(
            input_details[0]["index"],
            image_array
        )

        # Run prediction
        interpreter.invoke()

        # Get result
        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predicted_class = np.argmax(prediction[0])

        confidence = float(
            prediction[0][predicted_class]
        ) * 100

        result = classes[predicted_class]

        return render_template(
            "result.html",
            result=result,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
