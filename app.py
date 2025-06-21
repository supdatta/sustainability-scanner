from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import json
import io
from model import SustainabilityCNN

app = Flask(__name__)
CORS(app)

try:
    with open("sustainability_labels.json", "r") as f:
        label_facts = json.load(f)

    num_classes = len(label_facts)
    model = SustainabilityCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("sustainability_model.pt", map_location=torch.device("cpu")))
    model.eval()
    class_names = list(label_facts.keys())
except Exception as e:
    print("Error loading model:", e)
    model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return "Sustainability Scanner API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        fact = label_facts[label]

    return jsonify({
        'label': label,
        'fact': fact
    })

if __name__ == '__main__':
    app.run()
