from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class ModelType(str, Enum):
    BERT = "bert"
    LSTM = "lstm"
    GRU = "gru"

class PredictionResult(str, Enum):
    PHISHING = "phishing"
    LEGITIMATE = "legitimate"

class PredictionRequest(BaseModel):
    text: str = Field(..., description="The email content to analyze", min_length=1)
    model: ModelType = Field(..., description="The model to use for prediction")

class PredictionResponse(BaseModel):
    model: ModelType
    prediction: PredictionResult
    confidence: float = Field(..., ge=0.0, le=1.0)
