from flask import Flask, request, jsonify
import base64
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return "API is live"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")

        # Validate input
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        username = data.get("username")
        image_data = data.get("image")

        if not username or not image_data:
            return jsonify({"error": "Both 'username' and 'image' are required"}), 400

        # Decode the image to verify it's valid base64
        try:
            decoded_image = base64.b64decode(image_data)
            # You can now optionally save or process it
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400

        # Dummy result
        return jsonify({
            "username": username,
            "score": 91,
            "fact": "This looks like a sustainable environment!"
        })

    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
