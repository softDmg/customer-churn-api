# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def load_and_preprocess(
    input_path="data/raw/telco_churn.csv", 
    output_dir="data/processed"
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    # Drop ID and clean numeric
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Train/test split
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save processed files
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Saving to directory: {output_dir.resolve()}")
    print(f"Files expected: {list(output_dir.glob('*'))}")

    print(f"✅ Preprocessing complete. Files saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    load_and_preprocess(
        input_path="data/raw/telco_churn.csv",
        output_dir="data/processed"
    )