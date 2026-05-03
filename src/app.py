import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "models/delivery_time_model.pkl"

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="API for predicting food delivery time using a trained ML model",
    version="1.0.0"
)

# Load model
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# Request schema
class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    items_count: int = Field(..., gt=0)
    is_peak_hour: int = Field(..., ge=0, le=1)
    traffic_level: int = Field(..., ge=1, le=3)


# Response schema
class PredictionResponse(BaseModel):
    delivery_time_min: float


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Predict endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeliveryRequest):
    try:
        df = pd.DataFrame([{
            "distance_km": request.distance_km,
            "items_count": request.items_count,
            "is_peak_hour": request.is_peak_hour,
            "traffic_level": request.traffic_level
        }])

        pred = model.predict(df)[0]

        return {"delivery_time_min": round(float(pred), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))