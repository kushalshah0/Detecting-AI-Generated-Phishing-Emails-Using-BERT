import torch
import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.core.config import settings
from app.models import ModelType, PredictionResult
from app.services.preprocessor import Preprocessor
from app.services.shap_explainer import SHAPExplainer

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.models = {}
        self.tokenizers = {}
        self.preprocessor = Preprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.shap_explainers = {}
        self.initialized = True

    def load_models(self):
        """Loads all models and tokenizers."""
        print("Loading models...")
        
        # [BERT_RESTORE] Uncomment to enable BERT model
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

        # 2. Load RNN Tokenizer (Pickle)
        if os.path.exists(settings.TOKENIZER_PATH_RNN):
            try:
                print(f"Loading RNN Tokenizer from {settings.TOKENIZER_PATH_RNN}")
                with open(settings.TOKENIZER_PATH_RNN, 'rb') as f:
                    self.tokenizers['rnn'] = pickle.load(f)
                print("RNN Tokenizer loaded successfully.")
            except Exception as e:
                print(f"Error loading RNN Tokenizer: {e}")
        else:
            print(f"Warning: RNN Tokenizer file not found at {settings.TOKENIZER_PATH_RNN}")

        # 3. Load LSTM & GRU (PyTorch .pt files)
        rnn_models_to_load = {
            ModelType.LSTM: settings.MODEL_PATH_LSTM,
            ModelType.GRU: settings.MODEL_PATH_GRU
        }

        for model_type, path in rnn_models_to_load.items():
            if os.path.exists(path):
                try:
                    print(f"Loading {model_type} from {path}")
                    model = torch.load(path, map_location=self.device)
                    model.to(self.device)
                    model.eval()
                    self.models[model_type] = model
                    print(f"{model_type} loaded successfully.")
                except Exception as e:
                    print(f"Error loading {model_type}: {e}")
            else:
                print(f"Warning: {model_type} model file not found at {path}")

        print("Model loading process completed.")

        self._init_shap_explainers()

    def _init_shap_explainers(self):
        print("Initializing SHAP explainers...")
        for model_type, model in self.models.items():
            tokenizer = self.tokenizers.get('rnn')
            if tokenizer:
                try:
                    explainer = SHAPExplainer(model, tokenizer, self.device, model_type.value)
                    self.shap_explainers[model_type] = explainer
                    print(f"SHAP explainer initialized for {model_type}")
                except Exception as e:
                    print(f"Error initializing SHAP for {model_type}: {e}")

    def predict(self, text: str, model_type: ModelType) -> dict:
        # [BERT_RESTORE] Uncomment to enable BERT model
        # if model_type not in self.models and model_type == ModelType.BERT and 'bert' not in self.models:
        #      # Fallback logic for demo purposes if BERT failed to load or wasn't found
        #      # In a strict production environment, you might raise an error.
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
        #         # Assuming index 1 is 'phishing'
        #         phishing_prob = probs[0][1].item()
        # else:  # LSTM or GRU
        #     # [BERT_RESTORE] The above commented block should contain BERT prediction logic
        #     # When uncommenting BERT, wrap the code below in an else block

        model = self.models.get(model_type)
        tokenizer = self.tokenizers.get('rnn')

        if not model:
             # Fallback for demo
             print(f"{model_type} missing, returning mock prediction.")
             return {"prediction": PredictionResult.PHISHING, "confidence": 0.85}
        
        if not tokenizer:
             print("RNN Tokenizer missing, returning mock prediction")
             return {"prediction": PredictionResult.PHISHING, "confidence": 0.60}

        inputs = self.preprocessor.preprocess_rnn(text, tokenizer)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            output = model(inputs)
            # Handle different output shapes (sigmoid vs softmax)
            if output.shape[-1] == 1:
                phishing_prob = torch.sigmoid(output).item()
            else:
                probs = torch.softmax(output, dim=1)
                phishing_prob = probs[0][1].item()

        prediction = PredictionResult.PHISHING if phishing_prob > 0.5 else PredictionResult.LEGITIMATE
        confidence = phishing_prob if prediction == PredictionResult.PHISHING else (1 - phishing_prob)

        top_tokens = None
        if model_type in self.shap_explainers:
            try:
                top_tokens = self.shap_explainers[model_type].explain(text)
            except Exception as e:
                print(f"SHAP explanation error: {e}")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_tokens": top_tokens
        }

model_manager = ModelManager()
