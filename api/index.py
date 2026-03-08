import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MODEL_PATH_LSTM", os.getenv("MODEL_PATH_LSTM", "models_data/lstm_model.pt"))
os.environ.setdefault("MODEL_PATH_GRU", os.getenv("MODEL_PATH_GRU", "models_data/gru_model.pt"))
os.environ.setdefault("TOKENIZER_PATH_RNN", os.getenv("TOKENIZER_PATH_RNN", "models_data/rnn_tokenizer.pkl"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.model_manager import model_manager
from app.models import ModelType

app = FastAPI(title="Phishing Email Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str = Field(..., description="The email content to analyze", min_length=1)
    model: str = Field(..., description="Model to use: lstm or gru")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_email(request: PredictionRequest):
    try:
        model_type = ModelType(request.model)
        result = model_manager.predict(request.text, model_type)
        
        return {
            "model": request.model,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top_tokens": result.get("top_tokens")
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


model_manager.load_models()
