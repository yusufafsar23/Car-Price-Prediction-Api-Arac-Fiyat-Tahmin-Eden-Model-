# 🚗 Car Price Prediction API

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

---

## 🔹 Project Overview

This project is a **Car Price Prediction API** built using **FastAPI** and **professional machine learning pipelines**.  
The goal is to predict car prices based on features like **brand, year, mileage, fuel type, transmission, and condition**.

The machine learning model is implemented using **Gradient Boosting Regressor**, and the entire preprocessing and modeling workflow is handled through **scikit-learn pipelines**.  
- **Numerical features** are scaled using `StandardScaler`  
- **Categorical features** are encoded using `OneHotEncoder`  
- Hyperparameter optimization is performed with `GridSearchCV`  

This approach ensures a **robust and reproducible ML pipeline**, making the predictions more reliable and the code production-ready.  
Docker support is included to make deployment fast and consistent across different environments.

---

## 📊 Features

- Predict vehicle prices accurately based on car features  
- Fully pipeline-based machine learning workflow  
- Gradient Boosting Regressor with optimized hyperparameters  
- FastAPI backend with Swagger UI for testing  
- Dockerized for cross-platform deployment  

---

## 🛠 Tech Stack

- Python 3.11  
- FastAPI  
- Pandas & NumPy  
- Scikit-learn  
- Joblib  
- Docker  

---

## 🚀 Quick Start

### Clone the repository

```bash
git clone https://github.com/<your-username>/car-price-prediction-api.git
cd car-price-prediction-api

Install dependencies
pip install -r requirements.txt

Run locally with FastAPI
uvicorn main:app --reload

Open in browser: http://127.0.0.1:8000/
Swagger UI: http://127.0.0.1:8000/docs

Using Docker
docker build -t car-price-api .
docker run -d -p 8000:8000 car-price-api

Swagger UI: http://127.0.0.1:8000/docs

📌 API Endpoints
GET /
Check if the API is running:
{
  "message": "Car Price Prediction API is running!"
}
POST /predict
Predict the price of a car:
Request Example:

{
  "Brand": "Tesla",
  "Year": 2016,
  "Fuel_Type": "Petrol",
  "Transmission": "Manual",
  "Mileage": 114832,
  "Condition": "New"
}

Response Example:
{
  "Tahmin Edilen Fiyat": 26613.92
}

🗂 Project Structure
Car_Price_Predection/
│
├── main.py                     # FastAPI application
├── car_prediction_model.pkl     # Trained ML model
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # Project description
└── .gitignore                  # Files/folders to ignore
