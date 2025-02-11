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

# Define PyTorch model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# # Load PyTorch model
# model_path = "D:/FPTUniversity/Terminal4/DAP391m/Lab1/best_cnn.pt"

# if os.path.exists(model_path):
#     try:
#         model_best_cnn = SimpleCNN()
#         checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
#         model_best_cnn.load_state_dict(checkpoint['model'])
#         model_best_cnn.eval()
#         print("✅ PyTorch model loaded successfully!")
#     except Exception as e:
#         model_best_cnn = None
#         print(f"❌ Error loading PyTorch model: {e}")
# else:
#     model_best_cnn = None
#     print("❌ PyTorch model file not found!")

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
    elif model_type == 'torch':
        if model_best_cnn is None:
            return "Error: PyTorch model not loaded."

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(Image.open(image_path)).unsqueeze(0)

        with torch.no_grad():
            predictions = model_best_cnn(img_tensor).numpy()

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

# Real-time prediction for webcam images
def predict_animal_realtime(frame, model_type='keras'):
    """Predicts the animal in a real-time webcam frame."""
    img = cv2.resize(frame, (224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == 'keras':
        predictions = model_realtime.predict(img_array)
    elif model_type == 'torch':
        if model_best_cnn is None:
            return "Error: PyTorch model not loaded."

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            predictions = model_best_cnn(img_tensor).numpy()

    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

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
