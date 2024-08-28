
from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/model_inception.keras'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))  # Adjust size if necessary

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255.0  # Scaling
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    # Map prediction to class labels
    class_labels = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight",
        "Tomato__Tomato_mosaic_virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___Bacterial_spot"
    ]

    result = class_labels[preds[0]]
    return f"The Disease is {result}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    f = request.files['file']
    if f.filename == '':
        return redirect(request.url)
    if f:
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        file_path = os.path.join('uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        return render_template('index.html', result=preds)
    return None

if __name__ == '__main__':
    app.run(port=5001, debug=True)
