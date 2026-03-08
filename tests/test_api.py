import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.models import ModelType, PredictionResult


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    def test_predict_phishing_email_lstm(self):
        payload = {
            "text": "Your account has been compromised. Click here to reset your password immediately.",
            "model": "lstm"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "lstm"
        assert data["prediction"] in ["phishing", "legitimate"]
        assert 0 <= data["confidence"] <= 1
        assert "top_tokens" in data

    def test_predict_phishing_email_gru(self):
        payload = {
            "text": "Urgent! Your bank account has been hacked. Verify your identity now!",
            "model": "gru"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gru"
        assert data["prediction"] in ["phishing", "legitimate"]

    def test_predict_legitimate_email(self):
        payload = {
            "text": "Hi, just wanted to follow up on our meeting yesterday. Let me know if you have any questions.",
            "model": "lstm"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["phishing", "legitimate"]

    def test_predict_invalid_model(self):
        payload = {
            "text": "Test email content",
            "model": "invalid_model"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_empty_text(self):
        payload = {
            "text": "",
            "model": "lstm"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_model(self):
        payload = {
            "text": "Test email content"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestModelSchemas:
    def test_model_type_enum(self):
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.GRU.value == "gru"

    def test_prediction_result_enum(self):
        assert PredictionResult.PHISHING.value == "phishing"
        assert PredictionResult.LEGITIMATE.value == "legitimate"


class TestPreprocessor:
    def test_clean_text(self):
        from app.services.preprocessor import Preprocessor
        preprocessor = Preprocessor()
        
        text = "Click HERE http://example.com to WIN!!!"
        cleaned = preprocessor.clean_text(text)
        
        assert "http" not in cleaned
        assert cleaned == "click here to win"

    def test_preprocess_rnn(self):
        from app.services.preprocessor import Preprocessor
        preprocessor = Preprocessor()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
        
        result = preprocessor.preprocess_rnn("test text", mock_tokenizer)
        
        assert result.shape == (1, 150)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
