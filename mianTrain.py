import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded.')

# Function to get the class name based on class number
def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"

# Function to process the uploaded image and get the prediction result
def get_result(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.array(image) / 255.0  # Normalize image
    input_img = np.expand_dims(image, axis=0)
    probabilities = model.predict(input_img)[0]
    class_index = np.argmax(probabilities)
    return get_class_name(class_index)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        prediction = get_result(file_path)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
