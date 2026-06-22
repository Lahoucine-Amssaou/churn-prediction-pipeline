import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# 1. COLUMN DEFINITIONS

DROP_COLS = ["customerID"]  # unique ID, no predictive value

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

BINARY_COLS = ["SeniorCitizen"]

TARGET_COL = "Churn"

# 2. LOADING & CLEANING

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
  
    # We replace spaces with NaN, then fill with 0 (new customers have paid nothing)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    df = df.drop(columns=DROP_COLS)

    return df


# 3. PREPROCESSING PIPELINE

def build_preprocessor() -> ColumnTransformer:

    # StandardScaler: subtracts mean, divides by std deviation, it makes all numerical features live on the same scale
    numerical_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS)  # already 0/1, so no transformation needed
    ])

    return preprocessor

# 4. MAIN FUNCTION

def run_preprocessing(data_path: str, output_dir: str = "artifacts"):

    print("Loading and cleaning data...")
    df = load_and_clean(data_path)

    # Separate features from target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"Dataset shape: {X.shape}")
    print(f"Churn rate: {y.mean():.1%}")

    # Stratified split: ensures same churn ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 80% train, 20% test
        random_state=67,
        stratify=y            # preserve class balance in both splits
    )

    print(f"Train size: {len(X_train)} / Test size: {len(X_test)}")

    print("Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)  # only transform, never fit

    print(f"Processed feature shape: {X_train_processed.shape}")

    # Save artifacts so train.py and FastAPI can load them
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    with open(f"{output_dir}/train_test_data.pkl", "wb") as f:
        pickle.dump((X_train_processed, X_test_processed, y_train, y_test), f)

    print(f"Artifacts saved to {output_dir}/")
    return preprocessor, X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    run_preprocessing("data/telco_churn.csv")
