from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# Load label-to-score-and-fact mapping
with open("sustainability_labels.json", "r") as f:
    label_data = json.load(f)

# Define your model architecture (ResNet18-based)
from torchvision.models import resnet18

class SustainabilityCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SustainabilityCNN, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
model = SustainabilityCNN(num_classes=len(label_data))
checkpoint = torch.load("sustainability_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Define image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return jsonify({"message": "Sustainability API is live!"})

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files or "username" not in request.form:
        return jsonify({"error": "Missing image or username"}), 400

    image_file = request.files["image"]
    username = request.form["username"]

    try:
        # Decode image
        image = Image.open(image_file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_index = torch.argmax(outputs, dim=1).item()

        # Map to score and fact
        class_name = list(label_data.keys())[predicted_index]
        result = label_data[class_name]
        score = result["score"]
        fact = result["fact"]

        return jsonify({
            "message": f"Success for user {username}",
            "score": score,
            "fact": fact,
            "label": class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
