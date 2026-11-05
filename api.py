import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

app = FastAPI(title="Customer Churn API", version="1.0.0")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = None


class Prediction(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    label: str


@lru_cache(maxsize=1)
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")
    return joblib.load(MODEL_PATH)


def expected_columns(model) -> list:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(model.steps[0][1].feature_names_in_)


def to_frame(customer: Customer, columns: list) -> pd.DataFrame:
    row = customer.model_dump()
    if row["TotalCharges"] is None:
        row["TotalCharges"] = round(row["MonthlyCharges"] * row["tenure"], 2)
    df = pd.DataFrame([row]).replace(
        {"No internet service": "No", "No phone service": "No"}
    )
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"The model expects columns this API does not receive: {missing}")
    return df[columns]


def risk_level(p: float) -> str:
    return "Low" if p < 0.30 else "Medium" if p < 0.60 else "High"


@app.get("/")
def root():
    return {"message": "Customer Churn API is running", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        model = get_model()
        X = to_frame(customer, expected_columns(model))
        proba = float(model.predict_proba(X)[0, 1])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    pred = int(proba >= THRESHOLD)
    return Prediction(
        churn_prediction=pred,
        churn_probability=round(proba, 4),
        risk_level=risk_level(proba),
        label="Likely to churn" if pred else "Likely to stay",
    )