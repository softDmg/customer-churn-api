# src/train.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_model(use_smote=True):
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Optional: apply SMOTE to balance the training data
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("✅ SMOTE applied. New training set size:", X_train.shape)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model + scaler
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/logistic_model.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    print("✅ Model and scaler saved in /artifacts")

if __name__ == "__main__":
    train_model()