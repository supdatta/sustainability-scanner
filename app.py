from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import random
import os

app = Flask(__name__)
CORS(app)

model = None
class_names = None
facts = None
user_data_file = 'user_data.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model, class_names
    model_data = torch.load('sustainability_model.pt', map_location=device)
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(model_data['class_to_idx']))
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    class_names = {v: k for k, v in model_data['class_to_idx'].items()}

def load_facts():
    global facts
    with open('sustainability_facts.json') as f:
        facts = json.load(f)

def load_user_data():
    if os.path.exists(user_data_file):
        with open(user_data_file) as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open(user_data_file, 'w') as f:
        json.dump(data, f)

@app.route('/')
def home():
    return 'API is live'

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    try:
        img = Image.open(image.stream).convert('RGB')
    except:
        return jsonify({'error': 'Invalid image format'}), 400

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probs).item()
        predicted_class = class_names[predicted_idx]
        confidence = probs[predicted_idx].item()

    score = round(confidence * 100)
    fact = random.choice(facts[predicted_class]) if predicted_class in facts else "Be more sustainable!"

    username = request.form.get('username', 'guest')
    data = load_user_data()
    data.setdefault(username, []).append({'class': predicted_class, 'score': score})
    save_user_data(data)

    return jsonify({
        'class': predicted_class,
        'score': score,
        'fact': fact
    })

if __name__ == '__main__':
    load_model()
    load_facts()
    app.run(host='0.0.0.0', port=10000)
