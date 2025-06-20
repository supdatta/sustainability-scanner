from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import io
import os
import traceback

app = Flask(__name__)

# Load sustainability facts and labels
label_map = {}
facts_map = {}

try:
    with open("sustainability_labels.json", "r") as f:
        data = json.load(f)
        label_map = data
        facts_map = data  # If facts and labels are same file (key: fact), else split if needed
except Exception as e:
    print(" Failed to load sustainability_labels.json:", str(e))

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the full model object from .pt
model = None
model_loaded = False

try:
    model_bundle = torch.load("sustainability_model.pt", map_location="cpu")
    if isinstance(model_bundle, dict) and "model_state_dict" in model_bundle:
        # You trained it like this:
        # torch.save({'model_state_dict': model.state_dict(), 'class_to_idx': ...}, ...)
        base_model = models.resnet18(weights=None)
        num_classes = len(label_map)
        base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
        base_model.load_state_dict(model_bundle['model_state_dict'])
        model = base_model
    else:
        model = model_bundle  # You saved entire model via torch.save(model)
    
    model.eval()
    model_loaded = True
    print(" Model loaded successfully.")
except Exception as e:
    print(" Error loading model:", str(e))
    traceback.print_exc()

@app.route("/")
def index():
    return "🌿 Sustainability Classifier API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    try:
        image_tensor = transform(image).unsqueeze(0)  
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            label = list(label_map.keys())[predicted_idx]
            score = round(probs[0][predicted_idx].item() * 100, 2)

        fact = facts_map.get(label, "🌱 Sustainability is everyone's responsibility!")

        return jsonify({
            "label": label,
            "score": score,
            "fact": fact
        })

    except Exception as e:
        print(" Prediction error:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
