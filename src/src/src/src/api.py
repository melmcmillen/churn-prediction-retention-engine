from fastapi import FastAPI
import joblib
import pandas as pd
from .config import MODEL_PATH, SCALER_PATH

app = FastAPI()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.post("/predict_churn")
def predict_churn(customer: dict):
    df = pd.DataFrame([customer])
    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[0, 1]
    return {"churn_probability": float(prob)}
