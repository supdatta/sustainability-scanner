# Sustainability API 🚮🔍

A simple image classification API for sustainability analysis powered by a custom-trained CNN.

---

## 🚀 Getting Started

### 🧾 Prerequisites
- Python 3.10+
- `pip`, `virtualenv`, or Conda
- `archive.zip` (your dataset) unzipped into the `/dataset` folder

---

### 📦 Installation

```bash
git clone https://github.com/supdatta/sustainability_api.git
cd sustainability_api
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python3 train_model.py
```

This creates `sustainability_model.pt` used by the API.

---

## 🧪 Test Locally

Start the API:
```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) to check if it’s live.

Use `test_predict.py` to test predictions.

---

### 📤 Sample Prediction

```bash
curl -X POST http://localhost:5000/predict -F "file=@test.jpg"
```

Sample Response:
```json
{
  "prediction": "Biodegradable Waste",
  "score": 8,
  "fact": "This item decomposes naturally and safely."
}
```

---

## 🌍 Deployment (Railway)

1. Create a new project at [https://railway.app](https://railway.app)
2. Link this GitHub repository
3. Ensure `render.yaml` or equivalent config is present
4. Set build command:

   ```bash
   pip install -r requirements.txt
   ```

5. Set start command:

   ```bash
   gunicorn app:app
   ```

6. Add `runtime.txt` with content:

   ```
   python-3.10.12
   ```

---

## 📬 API Endpoints

| Method | Endpoint      | Description                          |
|--------|---------------|--------------------------------------|
| GET    | `/`           | Returns `"API is live"`              |
| POST   | `/predict`    | Returns prediction, score, and fact  |

---

## 🧾 Model Overview

CNN trained on a custom image dataset containing:

- Organic Waste  
- E-waste  
- Plastic  
- Metal  
- Paper  
- Non-Recyclable  
- Glass  
- Hazardous  
- Biodegradable  
- Other

---

## 🎯 Use Cases

- Mobile sustainability scanner apps  
- Waste classification systems  
- Educational environmental tools  
- Hackathons and AI demos  

---

## ⚠️ Notes

- Dataset not included in repo (add it manually)  
- Model must be trained using `train_model.py`  
- Ensure `sustainability_model.pt` exists before running `app.py`

---

## ©️ License

This project is developed for **educational and hackathon purposes** by **Saptangshu Datta**.
