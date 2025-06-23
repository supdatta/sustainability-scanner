# ♻️ Sustainability Image Classifier API

A Flask-based backend that classifies images into sustainable categories using a CNN model. This API powers mobile or web apps that need real-time sustainability scoring from images.

---

## 📦 Features

- 🔍 Predicts sustainability category from uploaded image
- 🧠 Uses custom-trained CNN (`sustainability_model.pt`)
- 📤 POST endpoint for prediction
- 🔐 Stores category-wise sustainability facts
- 🗃️ Tracks user scores in JSON

---

## 🛠️ Tech Stack

- **Backend**: Flask + Gunicorn
- **ML Framework**: PyTorch
- **Deployment**: Render / Railway (cloud-hosted)
- **Language**: Python 3.10.12
- **Data Storage**: JSON files

---

## 🗂️ Project Structure
```
├── dataset/ # Training images (not uploaded)
├── sustainability_model.pt # Trained PyTorch model
├── sustainability_labels.json # Label → Fact mapping
├── user_data.json # Stores scanned user scores
├── train_model.py # Model training script
├── predict_image.py # Predict single image
├── app.py # Flask API entry point
├── test_predict.py # Test client for /predict
├── model.py # CNN model class
├── requirements.txt # Dependencies
├── runtime.txt # Python version for Render
├── render.yaml # Render deploy config
├── .gitignore # Ignore datasets, temp files
```
---

##  Getting Started

### 🧾 Prerequisites

- Python 3.10+
- `pip`, `virtualenv` or Conda
- `archive.zip` (your dataset) unzipped to `/dataset`

---

###  Installation

```
git clone https://github.com/supdatta/sustainability_api.git
cd sustainability_api
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

```
### Train the Model
python3 train_model.py
This creates sustainability_model.pt used by the API.

### 🔬 Test Locally
python app.py
Open http://localhost:5000 to see API is live.


Use test_predict.py to test predictions.

### Sample Prediction
curl -X POST http://localhost:5000/predict \
-F "file=@test.jpg"
Response:

json
Copy
Edit
{
  "prediction": "Biodegradable Waste",
  "score": 8,
  "fact": "This item decomposes naturally and safely."
}

###🌍 Deployment (Railway)
Create new project at https://railway.app

Link GitHub repo

Ensure render.yaml or Railway equivalent config is present

Set build command:
pip install -r requirements.txt
Set start command:
gunicorn app:app
Add runtime.txt → python-3.10.12

###📬 API Endpoints
GET /
Returns: "API is live"


### POST /predict
Accepts: multipart/form-data image
Returns: Sustainability prediction, score, and fact

###🧠 Model Overview
CNN trained on custom image dataset containing:

Organic Waste
E-waste
Plastic
Metal
Paper
Non-Recyclable
Glass
Hazardous
Biodegradable
Other

###🎯 Use Cases
Mobile sustainability scanner apps

Waste classification systems

Educational environmental tools

Hackathons and AI demos

###⚠️ Notes
Dataset not included in repo (add it manually)

Model must be trained with train_model.py

Ensure sustainability_model.pt exists before running app.py

###©️ License
This project is developed for educational and hackathon purposes by Saptangshu Datta.
