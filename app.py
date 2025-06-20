from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import json

app = Flask(__name__)
CORS(app)

# Load model
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Load label -> score/fact map
with open("sustainability_labels.json") as f:
    label_map = json.load(f)

@app.route("/")
def home():
    return "🌱 Sustainability Scanner API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    username = data.get("username")
    image_data = data.get("image")

    if not username or not image_data:
        return jsonify({"error": "Missing username or image"}), 400

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform image (same as during training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        label_info = label_map.get(str(predicted_class), {
            "fact": "Unknown prediction",
            "score": 0
        })

        return jsonify({
            "message": f"Success for user {username}",
            "score": label_info["score"],
            "fact": label_info["fact"]
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
