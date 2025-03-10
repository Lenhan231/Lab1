from flask import Flask, render_template, request, jsonify
import os
import json
import base64
import google.generativeai as genai
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch.nn as nn

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load Keras models
try:
    model = load_model(r"Project\animal_model_best.h5")
    print("✅ Keras models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Keras models: {e}")

# Class labels
class_names = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# Configure Gemini API
try:
    with open(r"Project\api_key.json", "r") as file:
        data = json.load(file)
    genai.configure(api_key=data["key"])
    model_gemini = genai.GenerativeModel('gemini-pro')
    print("✅ Google Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")

# Image prediction function
def predict_animal(image_path, model_type='keras'):
    """Predicts the animal in an image using either Keras or PyTorch model."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == 'keras':
        predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Describe an animal using Gemini AI
def describe_animal(image_path):
    """Generates a description of the predicted animal using Gemini AI."""
    try:
        prompt = f"Describe the animal in the picture: {image_path}"
        chat = model_gemini.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text
    except:
        animal_class = predict_animal(image_path)
        prompt = f"Describe a {animal_class} in detail."
        chat = model_gemini.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image uploads and returns predictions & descriptions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    model_type = request.form.get('model_type', 'keras')
    animal = predict_animal(filepath, model_type)
    description = describe_animal(filepath)

    return jsonify({'animal': animal, 'description': description, 'image_path': filepath})

@app.route('/predict-realtime', methods=['POST'])
def predict_realtime():
    """Handles real-time webcam image classification."""
    data = request.json.get("image")
    if not data:
        return jsonify({'error': 'No image received'}), 400

    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    model_type = request.json.get('model_type', 'keras')
    animal = predict_animal_realtime(frame, model_type)

    return jsonify({'animal': animal})

if __name__ == '__main__':
    app.run(debug=True)
