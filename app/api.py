from fastapi import APIRouter, HTTPException
from app.models import PredictionRequest, PredictionResponse
from app.services.model_manager import model_manager

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_email(request: PredictionRequest):
    try:
        result = model_manager.predict(request.text, request.model)
        return PredictionResponse(
            model=request.model,
            prediction=result["prediction"],
            confidence=result["confidence"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log error in real app
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
