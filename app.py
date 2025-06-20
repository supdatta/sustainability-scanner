from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# Load label info
with open('sustainability_labels.json') as f:
    class_labels = json.load(f)

# Load sustainability facts
with open("sustainability_facts.json") as f:
    sustainability_facts = json.load(f)

# Load the full model
model = torch.load("sustainability_model.pt", map_location=torch.device('cpu'))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return "🌱 Sustainability Scanner API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_name = class_labels[str(class_idx)]
        score = int(outputs[0][class_idx].item() * 100)
        fact = sustainability_facts.get(class_name, "Make a green choice today!")

    return jsonify({
        "class": class_name,
        "score": min(max(score, 0), 100),
        "fact": fact
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
