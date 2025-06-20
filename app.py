from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os

app = Flask(__name__)
CORS(app)

# Load class labels
try:
    with open('sustainability_labels.json') as f:
        class_labels = json.load(f)
except FileNotFoundError:
    class_labels = {str(i): f"Category {i}" for i in range(10)}  # fallback

# Load model
try:
    model = torch.load("sustainability_model.pt", map_location=torch.device('cpu'))
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return "🌍 Sustainability API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on server."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img_file = request.files['image']
        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_label = class_labels.get(str(predicted_idx), f"Class {predicted_idx}")
            confidence = float(torch.softmax(output, dim=1)[0][predicted_idx]) * 100

        return jsonify({
            "class": predicted_label,
            "score": round(confidence),
            "fact": f"Learn more about {predicted_label} and its environmental impact!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
