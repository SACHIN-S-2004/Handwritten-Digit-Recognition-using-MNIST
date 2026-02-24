import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("mnist_cnn_model.keras")


def preprocess_image(img):
    """
    Convert image to:
    (1, 28, 28, 1)
    float32
    normalized (0–1)
    """

    img = img.convert("L")# To grayscale

    img = img.resize((28, 28))

    img_array = np.array(img)

    # img_array = 255 - img_array

    img_array = img_array.astype("float32") / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


@app.route("/")
def index():
    return render_template("index.html")


# 🔹 Canvas Prediction
@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    data = request.json["image"]

    # Decode base64 image
    image_data = base64.b64decode(data.split(",")[1])
    img = Image.open(BytesIO(image_data))

    processed = preprocess_image(img)

    prediction = model.predict(processed)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "digit": digit,
        "confidence": round(confidence * 100, 2)
    })


# 🔹 Upload Image Prediction
@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    file = request.files["file"]
    img = Image.open(file.stream)

    processed = preprocess_image(img)

    prediction = model.predict(processed)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return render_template("index.html",
        upload_digit=digit,
        upload_confidence=round(confidence * 100, 2))


if __name__ == "__main__":
    app.run(debug=True)