import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json
import base64
import os

# Load label map
with open("sustainability_labels.json", "r") as f:
    label_map = json.load(f)

# Load user data
if os.path.exists("user_data.json"):
    with open("user_data.json", "r") as f:
        user_data = json.load(f)
else:
    user_data = {}

# Define model architecture
class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SustainabilityCNN, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize model and load weights
model = SustainabilityCNN(num_classes=len(label_map))
model.load_state_dict(torch.load("sustainability_model.pt", map_location=torch.device("cpu")))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Sustainability Scanner API is live"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        username = data["username"]
        image_b64 = data["image"]

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = str(predicted.item())

        label = label_map.get(predicted_label, "Unknown")
        score = 100 if "green" in label.lower() or "recyclable" in label.lower() else 45

        fact = f"This looks like a {label}!"
        message = f"Success for user {username}"

        # Update user score
        if username not in user_data:
            user_data[username] = []
        user_data[username].append({"label": label, "score": score})

        with open("user_data.json", "w") as f:
            json.dump(user_data, f)

        return jsonify({
            "message": message,
            "score": score,
            "fact": fact
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
