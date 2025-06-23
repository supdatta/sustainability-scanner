# ♻️ Sustainability Scanner API

A lightweight backend for a **sustainability image classification** project, built using PyTorch and Flask. This backend powers a gamified mobile experience that lets users scan environments and receive instant sustainability feedback.

---

## 📁 Project Structure

.
├── pycache/ # Python bytecode cache
├── dataset/ # Folder to place unzipped image data
├── app.py # Main Flask API
├── app.py.save # Backup of app.py
├── model.py # CNN architecture
├── train_model.py # Training script for new dataset
├── predict_image.py # Standalone prediction script
├── test_predict.py # Local test script
├── sustainability_model.pt # Trained PyTorch model
├── sustainability_labels.json # Label-to-fact dictionary
├── user_data.json # Stores user scores and sessions
├── requirements.txt # Python package requirements
├── runtime.txt # Specifies Python version (3.10.12)
├── render.yaml # Render deployment config
├── .gitignore # Excludes files like archive.zip
├── README.md # This file

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
2️⃣ Prepare Dataset
Manually unzip archive.zip and structure your data like this:

css
Copy
Edit
dataset/
├── sustainable/
│   ├── 1.jpg
│   ├── 2.jpg
├── not_sustainable/
│   ├── a.jpg
│   ├── b.jpg
3️⃣ Train the Model
bash
Copy
Edit
python train_model.py
After training, it will save the model as sustainability_model.pt.

🌐 Run API Locally
bash
Copy
Edit
python app.py
You should see:
* Running on http://127.0.0.1:5000

🔌 API Endpoint
/predict (POST)
Request Body (JSON):

json
Copy
Edit
{
  "features": [0.25, 0.67, 0.91, ...]
}
Response:

json
Copy
Edit
{
  "prediction": ["sustainable"]
}
☁️ Deployment on Render
Push your project to GitHub.

Go to Render → New Web Service.

Connect your GitHub repo.

Set Build Command:

bash
Copy
Edit
pip install -r requirements.txt
Set Start Command:

bash
Copy
Edit
gunicorn app:app
Add a runtime.txt with:

Copy
Edit
python-3.10.12
🧠 Features
CNN-based sustainability image classifier

Flask API ready for mobile integration

Gamified backend with user score support

Easily retrainable with new data

🔧 Tech Stack
Python 3.10

Flask (API)

PyTorch (ML model)

Gunicorn (production server)

Render (cloud deployment)

✍️ Author
Made by @supdatta
For a sustainability-themed hackathon project 🌿

📌 Notes
sustainability_model.pt is the trained model file.

Use train_model.py to retrain if needed.

API works best when paired with a mobile scanner app.

