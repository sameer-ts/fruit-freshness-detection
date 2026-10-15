from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fruit_freshness_model.tflite")

classes = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "Please select an image."

    file = request.files["image"]

    if not file.filename:
        return "Please select an image."

    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((150, 150))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        input_info = input_details[0]
        input_dtype = input_info["dtype"]

        # Handle quantized integer model inputs.
        if np.issubdtype(input_dtype, np.integer):
            scale, zero_point = input_info["quantization"]

            if scale > 0:
                image_array = np.round(
                    image_array / scale + zero_point
                )

            limits = np.iinfo(input_dtype)
            image_array = np.clip(
                image_array, limits.min, limits.max
            ).astype(input_dtype)
        else:
            image_array = image_array.astype(input_dtype)

        interpreter.set_tensor(
            input_info["index"], image_array
        )
        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        ).flatten()

        print("Model output:", prediction)
        print("Output shape:", prediction.shape)

        if len(prediction) != len(classes):
            return (
                "Model output mismatch: expected 6 class scores, "
                f"but received {len(prediction)}. "
                "Check the model's output and training labels."
            ), 500

        predicted_class = int(np.argmax(prediction))
        result = classes[predicted_class]

        # Scores are displayed for diagnosis.
        scores = {
            name: round(float(score), 4)
            for name, score in zip(classes, prediction)
        }

        # Valid as a probability percentage only when the
        # model outputs probabilities, such as a softmax layer.
        confidence = float(prediction[predicted_class]) * 100

        return render_template(
            "result.html",
            result=result,
            confidence=round(confidence, 2),
            scores=scores
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return f"Prediction error: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
