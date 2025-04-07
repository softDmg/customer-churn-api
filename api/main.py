# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("artifacts/logistic_model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

# Initialize app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Define the input schema using Pydantic
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def welcome():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    label = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": label,
        "churn_probability": round(probability, 4)
    }