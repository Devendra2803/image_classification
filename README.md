# 🧠 Image Classification API (FastAPI + MobileNetV2)

This project is a FastAPI-based image classification API using **MobileNetV2** for retraining on custom image datasets. It supports ZIP upload of class-labeled images, retrains a MobileNetV2 model, and provides prediction via REST endpoints.

---

## 🚀 Features

- 📂 Upload and auto-extract ZIP image datasets  
- 🔁 Transfer learning using **MobileNetV2**  
- ✅ Support for multi-class classification  
- 📊 Displays training metrics: accuracy, precision, recall  
- 🧪 Predict image class using trained model  
- 🛠️ Built with **FastAPI**, **TensorFlow**, and **scikit-learn**

---

## 🗂️ Project Structure

image_classification/
│
├── app/
│ ├── main.py # FastAPI entrypoint
│ ├── mobilenet_processor.py # Training & prediction logic
│ ├── models.py # Pydantic models
│ └── database.py (optional) # DB logging (optional)
│
├── saved_model/ # Stores trained model
├── temp/ # Temporary ZIP extract and cleanup
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 🔧 Clone the Repository
```bash
git clone https://github.com/Devendra2803/image_classification.git
cd image_classification

📦 Install Dependencies
pip install -r requirements.txt

🚀 Run the FastAPI Server
uvicorn app.main:app --reload
Open your browser at: http://127.0.0.1:8000/docs

📬 API Endpoints
Method	Endpoint	Description
POST	/classifier/train	Upload ZIP & train model
POST	/classifier/predict	Upload image & get prediction

🗃️ Dataset Format (ZIP Structure)
The ZIP file should contain folders for each class label:
dataset.zip
├── cat/
│   ├── cat1.jpg
│   └── cat2.jpg
├── dog/
│   ├── dog1.jpg
│   └── dog2.jpg

📈 Sample Prediction Response
{
  "filename": "test.jpg",
  "predicted_class": "cat",
  "confidence": 0.9482
}
