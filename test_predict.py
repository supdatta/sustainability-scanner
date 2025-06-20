import requests
import base64

# Load image and convert to base64
with open("dataset/recyclable_materials/plastic/IMG_8674.jpeg", "rb") as img_file:
    b64_image = base64.b64encode(img_file.read()).decode('utf-8')

# Prepare payload
payload = {
    "username": "test_user",
    "image": b64_image
}

# Send request to your Flask API
response = requests.post("http://localhost:5000/predict", json=payload)

# Show result
print(response.json())
