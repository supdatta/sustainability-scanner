#♻️ Sustainability Scanner API
A lightweight backend built to classify images into sustainability categories using a custom-trained CNN model. Designed for integration with mobile apps or web-based scanners.

#📁 Project Structure
File / Folder	Purpose
app.py	Flask API endpoint for model inference
model.py	CNN architecture definition
train_model.py	Code to train the model using image dataset
predict_image.py	Utility to run prediction on a single image
test_predict.py	API test script using POST requests
sustainability_model.pt	Trained PyTorch model (binary or multiclass image classifier)
sustainability_labels.json	Label to fact mapping for frontend/game integration
user_data.json	Temporary user data storage (for tracking scores etc.)
dataset/	Training data folder (unzipped manually)
render.yaml	Render deployment config
runtime.txt	Python version lock for Render (3.10.12)
requirements.txt	Project dependencies (Flask, Torch, etc.)
.gitignore	Ignores large files like archive.zip and Python caches

# Getting Started
1. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
2. Train the model
Ensure your dataset is placed in dataset/ directory, structured like:

Copy
Edit
dataset/
├── sustainable/
│   ├── img1.jpg ...
├── not_sustainable/
│   ├── img2.jpg ...
Then run:

bash
Copy
Edit
python train_model.py
3. Start the API server
bash
Copy
Edit
python app.py
The server will start at http://localhost:5000/.

# Sample API Usage
Endpoint: /predict
Method: POST

Body (JSON):

json
Copy
Edit
{
  "features": [0.12, 0.34, ...]  // optional for feature-based input
}
or use multipart image input if extended.

Response:
json
Copy
Edit
{
  "prediction": ["sustainable"]
}

🌐 Deployment
This API is Render-compatible.

Steps to Deploy on Render:
Push to GitHub.

In Render, create a new Web Service.

Use render.yaml as the deploy config.

Set build command:

nginx
Copy
Edit
pip install -r requirements.txt
Start command:

nginx
Copy
Edit
gunicorn app:app

#📌 Notes
Model file is saved as: sustainability_model.pt

JSON label mappings allow for fact-based responses tied to predictions.

API designed to integrate with mobile gamified scanner apps.

🛠 Tech Stack
Python 3.10

Flask

PyTorch

Gunicorn

Render for deployment

👤 Authors
Developed by @supdatta
Built for use in a sustainability-focused hackathon
