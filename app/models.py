from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List

# class ModelType(str, Enum):
#     BERT = "bert"  # [BERT_RESTORE] Uncomment to enable BERT model
#     LSTM = "lstm"
#     GRU = "gru"

class ModelType(str, Enum):
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
    top_tokens: Optional[List[dict]] = None


class TokenScore(BaseModel):
    token: str
    shap_score: float
