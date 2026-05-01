"""
FastAPI application for Credit Card Fraud Detection.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import time
from loguru import logger
from contextlib import asynccontextmanager

from src.api.schemas import (
    TransactionInput, PredictionResponse,
    BatchTransactionInput, BatchPredictionResponse,
    ModelInfoResponse, RiskLevel
)
from src.models.predict import predict_single, predict_batch
from src.models.train import load_model

model_artifact = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection model...")
    try:
        model_artifact.update(load_model())
        logger.success(f"Model loaded: {model_artifact['model_name']}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="AI-powered fraud detection — single, batch, and CSV predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def build_response(result: dict) -> PredictionResponse:
    messages = {
        "LOW": "Transaction appears legitimate.",
        "MEDIUM": "Transaction requires monitoring.",
        "HIGH": "Potential fraud detected. Review recommended.",
        "CRITICAL": "High confidence fraud. Block immediately.",
    }
    return PredictionResponse(**result, message=messages.get(result["risk_level"], ""))


@app.get("/", tags=["Health"])
async def root():
    return {"status": "running", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy" if model_artifact else "degraded", "model_loaded": bool(model_artifact)}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if not model_artifact:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return ModelInfoResponse(
        model_name=model_artifact["model_name"],
        threshold=model_artifact["threshold"],
        feature_count=len(model_artifact["feature_names"]),
        status="loaded",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    try:
        result = predict_single(transaction.model_dump())
        return build_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_endpoint(batch: BatchTransactionInput):
    try:
        df = pd.DataFrame([t.model_dump() for t in batch.transactions])
        results_df = predict_batch(df)
        predictions = [
            PredictionResponse(
                is_fraud=bool(row["is_fraud"]),
                fraud_probability=row["fraud_probability"],
                risk_level=RiskLevel(row["risk_level"]),
                threshold_used=model_artifact.get("threshold", 0.5),
                model_used=model_artifact.get("model_name", "unknown"),
                message="Batch prediction"
            ) for _, row in results_df.iterrows()
        ]
        fraud_count = results_df["is_fraud"].sum()
        return BatchPredictionResponse(
            total=len(predictions), fraud_detected=int(fraud_count),
            fraud_rate=round(float(fraud_count / len(predictions)), 4),
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if len(df) > 5000:
            raise HTTPException(status_code=400, detail="CSV cannot exceed 5000 rows.")
        results_df = predict_batch(df)
        fraud_count = int(results_df["is_fraud"].sum())
        return {
            "filename": file.filename, "total_rows": len(df),
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / len(df), 4),
            "results": results_df[["fraud_probability", "is_fraud", "risk_level"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
