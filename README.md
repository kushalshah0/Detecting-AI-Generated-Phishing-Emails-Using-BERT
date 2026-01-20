# Phishing Email Detection API

A FastAPI-based REST API for detecting phishing emails using multiple AI models (BERT, LSTM, GRU).

## Features

- **Multi-Model Support**: Choose between BERT, LSTM, or GRU for predictions.
- **Efficient Loading**: Models are loaded once at startup.
- **Scalable**: built with FastAPI and Uvicorn.
- **Container Ready**: Easy to deploy on Docker/Cloud.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Models**
   The application expects model files in the `models_data/` directory by default. You can configure paths in `.env` or `app/core/config.py`.
   
   Structure:
   ```
   models_data/
   ├── bert_model/          # HuggingFace BERT model directory
   ├── lstm_model.pt        # PyTorch LSTM model
   ├── gru_model.pt         # PyTorch GRU model
   └── rnn_tokenizer.pkl    # Tokenizer for RNNs
   ```

3. **Run the Server**
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### Health Check
**GET** `/health`

Response:
```json
{"status": "healthy"}
```

### Predict
**POST** `/predict`

Payload:
```json
{
  "text": "Your account has been compromised. Click here to reset.",
  "model": "bert"  // Options: "bert", "lstm", "gru"
}
```

Response:
```json
{
  "model": "bert",
  "prediction": "phishing",
  "confidence": 0.98
}
```
