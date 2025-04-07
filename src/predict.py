# src/predict.py

import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("artifacts/logistic_model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

# Example input (replace these with real values later)
sample_input = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 2,
    "OnlineSecurity": 1,
    "OnlineBackup": 0,
    "DeviceProtection": 1,
    "TechSupport": 1,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5
}

def predict_churn(input_dict):
    df = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]  # probability of class 1 (churn)
    
    return prediction, probability

if __name__ == "__main__":
    prediction, prob = predict_churn(sample_input)
    label = "Churn" if prediction == 1 else "No Churn"
    print(f"🧠 Prediction: {label}")
    print(f"📊 Churn Probability: {prob:.2f}")