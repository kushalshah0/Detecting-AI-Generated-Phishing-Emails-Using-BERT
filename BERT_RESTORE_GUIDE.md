# BERT Restore Guide

This guide explains how to restore BERT model functionality that was commented out for deployment on resource-constrained environments like Render.

## Why BERT was Disabled

The BERT model requires significant memory resources (~500MB+ for the model + tokenizer) which exceeds the available memory on free Render tiers. The LSTM and GRU models are lightweight alternatives that work well within these constraints.

## Files to Modify

### 1. `.env`

Uncomment the BERT model path:

```bash
# Before:
# MODEL_PATH_BERT=kushalshah0/ai-generated-phishing-email-detection-bert  # [BERT_RESTORE] Uncomment to enable BERT model

# After:
MODEL_PATH_BERT=kushalshah0/ai-generated-phishing-email-detection-bert
```

### 2. `app/core/config.py`

Uncomment the BERT path setting:

```python
# Before:
# MODEL_PATH_BERT: str  # [BERT_RESTORE] Uncomment to enable BERT model

# After:
MODEL_PATH_BERT: str
```

### 3. `app/models.py`

Uncomment BERT in the ModelType enum:

```python
# Before:
class ModelType(str, Enum):
    LSTM = "lstm"
    GRU = "gru"

# After:
class ModelType(str, Enum):
    BERT = "bert"
    LSTM = "lstm"
    GRU = "gru"
```

### 4. `app/services/model_manager.py`

There are two sections to uncomment:

**a) Model Loading (around line 34-43):**

```python
# Before:
# # 1. Load BERT from Hugging Face Hub
# try:
#     print(f"Loading BERT from {settings.MODEL_PATH_BERT}")
#     self.tokenizers['bert'] = AutoTokenizer.from_pretrained(settings.MODEL_PATH_BERT)
#     bert_model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_PATH_BERT)
#     bert_model.to(self.device)
#     bert_model.eval()
#     self.models['bert'] = bert_model
#     print("BERT model loaded successfully.")
# except Exception as e:
#     print(f"Error loading BERT model: {e}")

# After:
# 1. Load BERT from Hugging Face Hub
try:
    print(f"Loading BERT from {settings.MODEL_PATH_BERT}")
    self.tokenizers['bert'] = AutoTokenizer.from_pretrained(settings.MODEL_PATH_BERT)
    bert_model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_PATH_BERT)
    bert_model.to(self.device)
    bert_model.eval()
    self.models['bert'] = bert_model
    print("BERT model loaded successfully.")
except Exception as e:
    print(f"Error loading BERT model: {e}")
```

**b) Prediction Logic (around line 82-120):**

```python
# Before:
# if model_type not in self.models and model_type == ModelType.BERT and 'bert' not in self.models:
#      print("BERT model missing/failed to load, using mock response for demo.")
#      return {"prediction": PredictionResult.LEGITIMATE, "confidence": 0.999}

# if model_type == ModelType.BERT:
#     model = self.models.get('bert')
#     tokenizer = self.tokenizers.get('bert')
    
#     if not model or not tokenizer:
#         raise ValueError("BERT model or tokenizer is not loaded.")
    
#     input_ids, attention_mask = self.preprocessor.preprocess_bert(text, tokenizer)
#     input_ids = input_ids.to(self.device)
#     attention_mask = attention_mask.to(self.device)
    
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#         probs = torch.softmax(outputs.logits, dim=1)
#         phishing_prob = probs[0][1].item()
# else:  # LSTM or GRU

# After:
if model_type not in self.models and model_type == ModelType.BERT and 'bert' not in self.models:
     print("BERT model missing/failed to load, using mock response for demo.")
     return {"prediction": PredictionResult.LEGITIMATE, "confidence": 0.999, "top_tokens": None}

if model_type == ModelType.BERT:
    model = self.models.get('bert')
    tokenizer = self.tokenizers.get('bert')
    
    if not model or not tokenizer:
        raise ValueError("BERT model or tokenizer is not loaded.")
    
    input_ids, attention_mask = self.preprocessor.preprocess_bert(text, tokenizer)
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        phishing_prob = probs[0][1].item()
        
    prediction = PredictionResult.PHISHING if phishing_prob > 0.5 else PredictionResult.LEGITIMATE
    confidence = phishing_prob if prediction == PredictionResult.PHISHING else (1 - phishing_prob)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_tokens": None  # SHAP not implemented for BERT in this version
    }
else:  # LSTM or GRU
```

**Important:** When uncommenting BERT prediction, you need to wrap the existing LSTM/GRU code in an `else` block.

## Deployment Considerations

To run BERT on Render, you need:

1. **Upgrade to a paid tier** - The free tier has ~512MB RAM which is insufficient
2. **Use a smaller BERT model** - Consider `distilbert-base-uncased` instead of full BERT
3. **Enable GPU** - Add a GPU to your Render service for better performance
4. **Increase timeout** - Model loading may take 30+ seconds on first request

## Quick Search for [BERT_RESTORE] Markers

To find all locations that need modification:

```bash
grep -rn "BERT_RESTORE" .
```

This will show you all the commented sections that need to be restored.
