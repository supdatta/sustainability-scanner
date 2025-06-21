from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes):
        super(SustainabilityCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open('sustainability_labels.json', 'r') as f:
        label_facts = json.load(f)
    num_classes = len(label_facts)
    model = SustainabilityCNN(num_classes)
    model.to(device)
    checkpoint = torch.load('sustainability_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    model_loaded = True
except Exception as e:
    model_loaded = False
    model = None
    load_error = str(e)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return "API is live"

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded', 'details': load_error}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(request.files['image']).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, 1).item()
        label = idx_to_class[predicted]
        fact = label_facts.get(label, "No fact available.")
    return jsonify({
        'label': label,
        'score': predicted,
        'fact': fact
    })
