import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)

# VALID CUSTOMER PAYLOAD

VALID_CUSTOMER = {
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


# TEST 1: HEALTH CHECK

def test_root_endpoint():
    """API should respond to GET / with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


# TEST 2: VALID PREDICTION

def test_predict_valid_input():
    """
    A well-formed request should return status 200
    and a response with the expected structure.
    """
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    data = response.json()

    # We check if all expected fields are present
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data

    # We check types and value ranges
    assert isinstance(data["churn_probability"], float)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_level"] in ["low", "medium", "high"]


# TEST 3: MISSING FIELD

def test_predict_missing_field():
    """
    A request missing a required field should return 422 (Unprocessable Entity)
    not a 500 crash. Pydantic handles this automatically.
    """
    incomplete_customer = VALID_CUSTOMER.copy()
    del incomplete_customer["Contract"]  # remove a required field

    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422


# TEST 4: WRONG DATA TYPE

def test_predict_wrong_type():
    """
    Sending a string where a number is expected should return 422,
    not a 500 crash.
    """
    bad_customer = VALID_CUSTOMER.copy()
    bad_customer["tenure"] = "two months"  # should be an integer

    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


# TEST 5: PROBABILITY RANGE

def test_probability_is_valid():
    """
    The churn_probability must always be between 0 and 1.
    This catches any bug in the predict_proba extraction logic.
    """
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


# TEST 6: RISK LEVEL CONSISTENCY

def test_risk_level_matches_probability():
    """
    The risk_level label must be consistent with the churn_probability value.
    Tests the business logic in get_risk_level().
    """
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    data = response.json()
    prob = data["churn_probability"]
    risk = data["risk_level"]

    if prob >= 0.7:
        assert risk == "high"
    elif prob >= 0.4:
        assert risk == "medium"
    else:
        assert risk == "low"
