from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import json
import os
from predict_image import predict_image

app = Flask(__name__)

with open("sustainability_labels.json") as f:
    material_data = json.load(f)

if not os.path.exists("user_data.json"):
    with open("user_data.json", "w") as f:
        json.dump({}, f)

@app.route("/")
def home():
    return "Sustainability Scanner API is live"

@app.route("/predict", methods=["POST"])
def predict():
    body = request.json
    image_b64 = body.get("image")
    username = body.get("username")

    if not image_b64 or not username:
        return jsonify({"error": "Missing image or username"}), 400

    try:
        img_bytes = base64.b64decode(image_b64)
        label = predict_image(img_bytes)
        info = material_data.get(label, {
            "score": 5,
            "fact": "No sustainability info available."
        })

        with open("user_data.json") as f:
            users = json.load(f)

        user = users.get(username, {"total_score": 0, "scans": []})
        user["total_score"] += info["score"]
        user["scans"].append({"item": label, "score": info["score"]})
        users[username] = user

        with open("user_data.json", "w") as f:
            json.dump(users, f, indent=2)

        return jsonify({
            "item": label,
            "score": info["score"],
            "fact": info["fact"],
            "total_score": user["total_score"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/profile/<username>", methods=["GET"])
def profile(username):
    with open("user_data.json") as f:
        users = json.load(f)

    if username not in users:
        return jsonify({"error": "User not found"}), 404

    return jsonify(users[username])

if __name__ == "__main__":
    app.run(debug=True)
