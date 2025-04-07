# 🧠 Customer Churn Prediction API

A machine learning-powered API that predicts customer churn using logistic regression, built with FastAPI and deployed via Docker on Render.

## 🚀 Live Demo

👉 [Try it live](https://customer-churn-api-xmv5.onrender.com/docs) – Swagger UI

## 🔍 What It Does

- Predicts whether a customer is likely to churn or stay
- Returns churn probability and prediction label
- Exposes an easy-to-use REST API

## 📊 Problem Domain

Telco companies lose revenue when customers churn. This app helps predict churn using features like contract type, tenure, monthly charges, and more — allowing for proactive retention strategies.

## 🧱 Tech Stack

- **Model**: Logistic Regression (Scikit-learn)
- **API**: FastAPI
- **Deployment**: Docker + Render
- **Preprocessing**: Pandas, SMOTE
- **Serialization**: joblib

## ��️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/softDmg/customer-churn-api.git
cd customer-churn-api

# Build the Docker image
docker build -t churn-api .

# Run the container
docker run -p 8000:8000 churn-api

# Visit the docs
http://localhost:8000/docs
