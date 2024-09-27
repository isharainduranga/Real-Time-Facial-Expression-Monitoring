import base64
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, jsonify
from flask_cors import CORS  # If you are using this across domains
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model
model = models.resnet18(weights=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Assuming 4 emotion classes

# Load the model with error handling
try:
    model.load_state_dict(torch.load('model.pth'))
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    exit(1)

model.eval()

# Emotion mapping
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Sad', 3: 'Fear'}

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load OpenCV's pre-trained face detector (using Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the face cascade loaded correctly
if face_cascade.empty():
    print("Error loading Haar Cascade Classifier.")
    exit(1)

def preprocess_image(image):
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    data = request.json
    image_data = data['image']

    # Decode the base64 image data
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes))

    # Convert image to a format OpenCV can work with
    opencv_image = np.array(img.convert('RGB'))
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Optionally resize image for more consistent face detection
    if opencv_image.shape[1] > 1000:
        opencv_image = cv2.resize(opencv_image, (1000, int(opencv_image.shape[0] * 1000 / opencv_image.shape[1])))

    # Detect faces
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No faces detected"}), 400

    results = []

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        face = opencv_image[y:y+h, x:x+w]  # Crop the face
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL image

        # Preprocess the face for the model
        face_tensor = preprocess_image(face_pil)

        # Predict emotion for the face
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted_class = torch.max(outputs, 1)
            predicted_emotion = emotion_map[predicted_class.item()]

        # Store the coordinates and emotion
        results.append({
            'emotion': predicted_emotion,
            'coordinates': [int(x), int(y), int(w), int(h)]
        })

    # Return the emotions and coordinates as a JSON response
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
