from flask import Flask, render_template, request, jsonify
import os
import json
import base64
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load models
model = load_model(r"D:\GitHub\Lab1\best_cnn.pt")
model_realtime = load_model(r"animal_recognition_model.h5")
class_names = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# Configure Gemini API
with open(r"D:\FPTUniversity\Terminal4\DAP391m\Lab1\Project\api_key.json", "r") as file:
    data = json.load(file)

genai.configure(api_key=data["key"])
model_gemini = genai.GenerativeModel('gemini-pro')

def predict_animal(image_path):
    """Predict the animal in the uploaded image."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

def describe_animal(image_path):
    """Describe the animal using Gemini AI."""
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

def predict_animal_realtime(frame):
    """Predict the animal in a real-time webcam frame."""
    img = cv2.resize(frame, (224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model_realtime.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    animal = predict_animal(filepath)
    description = describe_animal(filepath)

    return jsonify({'animal': animal, 'description': description, 'image_path': filepath})

@app.route('/predict-realtime', methods=['POST'])
def predict_realtime():
    data = request.json.get("image")
    if not data:
        return jsonify({'error': 'No image received'}), 400

    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    animal = predict_animal_realtime(frame)
    
    return jsonify({'animal': animal})

if __name__ == '__main__':
    app.run(debug=True)
