from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import io
import random

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes):
        super(SustainabilityCNN, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = None
class_to_idx = None
idx_to_class = None
facts = {}

try:
    with open("sustainability_labels.json") as f:
        facts = json.load(f)
except:
    facts = {}

try:
    checkpoint = torch.load("sustainability_model.pt", map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = SustainabilityCNN(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Sustainability API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        label_idx = predicted.item()
        label = idx_to_class[label_idx]
        score = random.randint(45, 100)
        fact = facts.get(label, "This is an interesting item for sustainability!")
        return jsonify({'label': label, 'score': score, 'fact': fact})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
