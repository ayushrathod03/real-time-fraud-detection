import time
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.schemas import TransactionRequest, PredictionResponse

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    ml_artifacts["model"] = joblib.load("models/xgb_fraud.joblib")
    ml_artifacts["scaler"] = joblib.load("models/scaler.joblib")
    ml_artifacts["features"] = joblib.load("models/features.joblib")
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[STARTUP] Loaded XGBoost artifacts in {load_ms:.2f} ms")
    yield
    ml_artifacts.clear()


app = FastAPI(title="Real-Time Fraud Detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml_artifacts}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest):
    t0 = time.perf_counter()
    payload = tx.model_dump()
    row = np.array([[payload[f] for f in ml_artifacts["features"]]], dtype=np.float64)
    row_scaled = ml_artifacts["scaler"].transform(row)
    prob = float(ml_artifacts["model"].predict_proba(row_scaled)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=round(prob, 6),
        is_fraud=prob >= 0.5,
        model_version="xgb-v1.0",
        inference_latency_ms=round(latency_ms, 3),
    )