# streamlit/app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("artifacts/logistic_model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

# App config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📞 Customer Churn Prediction")
st.markdown("Use the form below to check if a customer is likely to churn.")

st.markdown("---")
st.subheader("👤 Customer Profile")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Has Partner", [0, 1])
    Dependents = st.selectbox("Has Dependents", [0, 1])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    Contract = st.selectbox("Contract Type", [0, 1, 2], format_func=lambda x: ["Month-to-Month", "One Year", "Two Year"][x])
    PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
    PaymentMethod = st.selectbox("Payment Method", [0, 1, 2, 3])
    MonthlyCharges = st.slider("Monthly Charges ($)", 10, 120, 70)
    TotalCharges = st.slider("Total Charges ($)", 0, 9000, 800)

st.markdown("---")
st.subheader("💻 Services")

col3, col4 = st.columns(2)
with col3:
    PhoneService = st.selectbox("Phone Service", [0, 1])
    MultipleLines = st.selectbox("Multiple Lines", [0, 1])
    InternetService = st.selectbox("Internet Service", [0, 1, 2], format_func=lambda x: ["DSL", "Fiber", "None"][x])
    OnlineSecurity = st.selectbox("Online Security", [0, 1])

with col4:
    OnlineBackup = st.selectbox("Online Backup", [0, 1])
    DeviceProtection = st.selectbox("Device Protection", [0, 1])
    TechSupport = st.selectbox("Tech Support", [0, 1])
    StreamingTV = st.selectbox("Streaming TV", [0, 1])
    StreamingMovies = st.selectbox("Streaming Movies", [0, 1])

# Collect inputs
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

# Predict
st.markdown("---")
if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        try:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            label = "Churn" if prediction == 1 else "No Churn"

            st.markdown(f"### 🧠 Prediction: **{label}**")
            st.markdown(f"**Churn Probability:** `{probability * 100:.2f}%`")

            if prediction == 1:
                st.warning("⚠️ Customer is likely to churn.")
            else:
                st.success("✅ Customer is likely to stay.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Hide footer & menu
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)