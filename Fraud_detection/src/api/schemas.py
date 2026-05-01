"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionInput(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {"example": {
            "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            "Amount": 149.62, "Time": 0.0
        }}


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    threshold_used: float
    model_used: str
    message: str


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

    @validator("transactions")
    def check_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000.")
        return v


class BatchPredictionResponse(BaseModel):
    total: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    feature_count: int
    status: str
