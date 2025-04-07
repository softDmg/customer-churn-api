# app.py

import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📞 Customer Churn Prediction")
st.markdown("Enter customer info to predict churn risk:")

# Collect user inputs
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", [0, 1])
Dependents = st.selectbox("Has Dependents", [0, 1])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", [0, 1])
MultipleLines = st.selectbox("Multiple Lines", [0, 1])
InternetService = st.selectbox("Internet Service", [0, 1, 2], format_func=lambda x: ["DSL", "Fiber", "None"][x])
OnlineSecurity = st.selectbox("Online Security", [0, 1])
OnlineBackup = st.selectbox("Online Backup", [0, 1])
DeviceProtection = st.selectbox("Device Protection", [0, 1])
TechSupport = st.selectbox("Tech Support", [0, 1])
StreamingTV = st.selectbox("Streaming TV", [0, 1])
StreamingMovies = st.selectbox("Streaming Movies", [0, 1])
Contract = st.selectbox("Contract Type", [0, 1, 2], format_func=lambda x: ["Month-to-Month", "One Year", "Two Year"][x])
PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
PaymentMethod = st.selectbox("Payment Method", [0, 1, 2, 3])
MonthlyCharges = st.slider("Monthly Charges ($)", 10, 120, 70)
TotalCharges = st.slider("Total Charges ($)", 0, 9000, 800)

# Prepare input dictionary
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# Button to send data
if st.button("Predict Churn"):
    with st.spinner("Sending data to model..."):
        url = "http://localhost:8000/predict"
        # url = "https://customer-churn-api-xmv5.onrender.com/predict"
        try:
            response = requests.post(url, json=input_data)
            result = response.json()

            st.success(f"Prediction: **{result['prediction']}**")
            st.info(f"Churn Probability: `{result['churn_probability'] * 100:.2f}%`")

        except Exception as e:
            st.error(f"Error: {e}")