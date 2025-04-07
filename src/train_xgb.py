# src/train_xgb.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

def train_model(use_smote=True):
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Apply SMOTE
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("✅ SMOTE applied. Balanced training set:", X_train.shape)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("🏁 ROC AUC:", roc_auc_score(y_test, y_prob))

    # Save model + scaler
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/xgb_model.joblib")
    joblib.dump(scaler, "artifacts/xgb_scaler.joblib")
    print("✅ Model and scaler saved to /artifacts")

if __name__ == "__main__":
    train_model()