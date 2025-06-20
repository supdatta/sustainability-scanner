from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
import pickle
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
with open("sustainability_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label descriptions
with open("sustainability_labels.json", "r") as f:
    label_info = json.load(f)

# Optional: load or initialize user data
USER_DATA_FILE = "user_data.json"
if os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "r") as f:
        user_data = json.load(f)
else:
    user_data = {}

# Helper to decode base64 image
def decode_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((64, 64))  # ensure same size as model training
    return np.array(image).reshape(1, 64, 64, 3) / 255.0

# API root
@app.route("/")
def home():
    return "✅ Sustainability API is Live"

# /predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check for required keys
        if "username" not in data or "image" not in data:
            return jsonify({"error": "Missing username or image"}), 400

        username = data["username"]
        img_array = decode_image(data["image"])

        # Get model prediction
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_label = str(predicted_index)

        # Get label info
        label = label_info.get(predicted_label, {}).get("label", f"Class {predicted_label}")
        score = label_info.get(predicted_label, {}).get("score", 50)
        fact = label_info.get(predicted_label, {}).get("fact", "No fact available.")

        # Update user score
        user_score = user_data.get(username, 0) + score
        user_data[username] = user_score

        # Save updated user data
        with open(USER_DATA_FILE, "w") as f:
            json.dump(user_data, f)

        return jsonify({
            "username": username,
            "label": label,
            "score_awarded": score,
            "total_score": user_score,
            "fact": fact
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
