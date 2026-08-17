import os

# Limit TensorFlow CPU usage before importing TensorFlow
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Limit TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Model
MODEL_PATH = "BrainTumor10EpochsCategorical.h5"

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

# Warm up the model once
try:
    dummy_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    model(dummy_image, training=False)
    print("Model warm-up completed.")
except Exception as e:
    print("Model warm-up error:", e)


def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"
    else:
        return "Unknown"


def get_result(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    image = cv2.resize(image, (64, 64))

    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    input_img = np.expand_dims(image, axis=0)

    print("Starting prediction...")

    # Direct TensorFlow inference
    output = model(input_img, training=False)

    probabilities = output.numpy()[0]

    class_index = int(np.argmax(probabilities))

    prediction = get_class_name(class_index)

    print("Prediction:", prediction)

    return prediction


UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    if file:

        filename = secure_filename(file.filename)

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(file_path)

        try:
            prediction = get_result(file_path)

            return render_template(
                "result.html",
                prediction=prediction
            )

        except Exception as e:
            print("Prediction error:", e)
            return f"Prediction failed: {str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
