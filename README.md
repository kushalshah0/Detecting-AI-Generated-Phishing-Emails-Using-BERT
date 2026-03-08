# Phishing Email Detection API

A FastAPI-based REST API for detecting AI-generated phishing emails using deep learning models (LSTM, GRU) with SHAP explainability.

## Features

- **Multi-Model Support**: Choose between LSTM or GRU models for predictions
- **SHAP Explainability**: Get detailed token-level attribution showing which words contributed to the prediction
- **Efficient Loading**: Models loaded once at startup with lazy initialization
- **Production Ready**: Built with FastAPI and Uvicorn, deployable on cloud platforms
- **Resource Optimized**: Lightweight models (~27MB) optimized for free-tier deployment (Render)

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **ML Models**: PyTorch (LSTM, GRU)
- **Explainability**: SHAP (KernelExplainer)
- **Deployment**: Render, Docker

## Setup

### Prerequisites

- Python 3.10+
- PyTorch

### Installation

```bash
pip install -r requirements.txt
```

### Model Files

Place your trained model files in `models_data/`:

```
models_data/
├── lstm_model.pt        # PyTorch LSTM model
├── gru_model.pt        # PyTorch GRU model
└── rnn_tokenizer.pkl   # Tokenizer for RNNs
```

Configure paths in `.env` or `app/core/config.py`.

### Run Locally

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

**GET** `/health`

```json
{"status": "healthy"}
```

### Predict

**POST** `/predict`

Request:

```json
{
  "text": "Your account has been compromised. Click here to reset your password immediately.",
  "model": "lstm"
}
```

Response:

```json
{
  "model": "lstm",
  "prediction": "phishing",
  "confidence": 0.94,
  "top_tokens": [
    {"token": "click", "shap_score": 0.15},
    {"token": "compromised", "shap_score": 0.12},
    {"token": "immediately", "shap_score": 0.08},
    {"token": "reset", "shap_score": 0.05},
    {"token": "password", "shap_score": 0.03}
  ]
}
```

**Parameters:**
- `text` (required): Email content to analyze
- `model` (required): Model type - `"lstm"` or `"gru"`

**Response:**
- `model`: The model used for prediction
- `prediction`: `"phishing"` or `"legitimate"`
- `confidence`: Prediction confidence (0-1)
- `top_tokens`: List of tokens with SHAP attribution scores

## Deployment

### Vercel (Recommended)

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel login`
3. Deploy: `vercel --prod`

Or connect your GitHub repo to Vercel for automatic deployments.

Environment Variables (configure in Vercel Dashboard):
- `MODEL_PATH_LSTM` = `models_data/lstm_model.pt`
- `MODEL_PATH_GRU` = `models_data/gru_model.pt`
- `TOKENIZER_PATH_RNN` = `models_data/rnn_tokenizer.pkl`

### Render

1. Push code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Project Structure

```
.
├── api/
│   └── index.py              # Vercel serverless function
├── app/
│   ├── api.py              # API routes
│   ├── models.py           # Pydantic schemas
│   ├── core/
│   │   └── config.py       # Configuration
│   └── services/
│       ├── model_manager.py    # Model loading & inference
│       ├── shap_explainer.py   # SHAP explainer
│       └── preprocessor.py     # Text preprocessing
├── models_data/            # Model files
├── main.py                 # Application entry point
├── vercel.json             # Vercel configuration
├── render.yaml             # Render deployment config
├── requirements.txt        # Python dependencies
└── runtime.txt             # Python version
```

## License

MIT
