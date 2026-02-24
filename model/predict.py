import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mnist_cnn_model.keras")


def preprocess_image(image_path):
    """
    Converts image to:
    (1, 28, 28, 1)
    float32
    normalized (0-1)
    """

    img = Image.open(image_path)

    img = img.convert("L") # to grayscale

    img = img.resize((28, 28))

    img_array = np.array(img)

    # Invert if background is white
    # img_array = 255 - img_array

    img_array = img_array.astype("float32") / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


def predict_digit(image_path):
    processed_image = preprocess_image(image_path)

    prediction = model.predict(processed_image)

    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    return digit, confidence


if __name__ == "__main__":
    image_path = "mnist-example.png"

    digit, confidence = predict_digit(image_path)

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.4f}")