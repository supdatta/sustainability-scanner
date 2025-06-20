from flask import Flask, request, jsonify
import logging
import os
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

        # Check if JSON is present
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        # Check for required fields
        if "username" not in data or "image" not in data:
            return jsonify({"error": "Missing 'username' or 'image' in request"}), 400

        username = data["username"]
        image_data = data["image"]

        # TODO: Decode image, run model, return prediction
        return jsonify({
            "message": f"Success for user {username}",
            "score": 85,
            "fact": "This looks like a very green image!"
        })

    except Exception as e:
        app.logger.error(f"Exception: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
