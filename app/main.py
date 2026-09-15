import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os

# 1. LOAD ARTIFACTS

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

try:
    with open(f"{ARTIFACTS_DIR}/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(f"{ARTIFACTS_DIR}/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("✓ Model and preprocessor loaded successfully")

except FileNotFoundError as e:
    print(f"✗ Artifact not found: {e}")
    print("  Run src/train.py first to generate artifacts/")
    preprocessor = None
    model = None


# 2. FASTAPI APP INSTANCE

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability for a telecom company.",
    version="1.0.0"
)


# 3. INPUT SCHEMA (PYDANTIC)

class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months as a customer")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge in USD")
    TotalCharges: float = Field(..., ge=0, description="Total amount charged")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, 0 otherwise")
    gender: str = Field(..., description="Male or Female")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 2,
                "MonthlyCharges": 85.5,
                "TotalCharges": 171.0,
                "SeniorCitizen": 0,
                "gender": "Female",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }


# 4. OUTPUT SCHEMA

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn between 0 and 1")
    churn_prediction: bool = Field(..., description="True if predicted to churn")
    risk_level: str = Field(..., description="low / medium / high based on probability")


# 5. HELPER: RISK LEVEL

def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    else:
        return "low"


# 6. ENDPOINTS

@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "api": "Churn Prediction API v1.0.0"
    }


@app.get("/health")
def health():
    """
    Detailed health check.
    CI/CD and monitoring tools ping this endpoint to verify the service is up.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run training first."
        )
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Main prediction endpoint.
    Accepts customer features, returns churn probability and risk level.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first."
        )

    input_data = pd.DataFrame([customer.dict()])

    # We apply the exact same preprocessing used during training
    try:
        input_processed = preprocessor.transform(input_data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Preprocessing failed: {str(e)}"
        )

    proba_output = model.predict_proba(input_processed)[0]
    churn_proba = float(proba_output[1] if len(proba_output) > 1 else proba_output[0])

    return PredictionResponse(
        churn_probability=round(churn_proba, 4),
        churn_prediction=churn_proba >= 0.5,
        risk_level=get_risk_level(churn_proba)
    )
