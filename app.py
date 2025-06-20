from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# Load sustainability labels
try:
    with open('sustainability_labels.json') as f:
        class_labels = json.load(f)
except FileNotFoundError:
    class_labels = {str(i): f"Class {i}" for i in range(10)}  # fallback labels

# Define model architecture (same as used in train_model.py)
class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SustainabilityCNN, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model and weights
try:
    model = SustainabilityCNN(num_classes=len(class_labels))
    state_dict = torch.load("sustainability_model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return "✅ Sustainability Scanner API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_label = class_labels.get(str(predicted_idx), f"Class {predicted_idx}")
            confidence = float(torch.softmax(output, dim=1)[0][predicted_idx]) * 100

        return jsonify({
            "class": predicted_label,
            "score": round(confidence),
            "fact": f"Fun Fact: {predicted_label} impacts the environment in unique ways!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
