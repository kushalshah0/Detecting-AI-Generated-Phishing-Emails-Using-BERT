import torch
import numpy as np
import shap
from typing import List, Dict
from app.services.preprocessor import Preprocessor


class SHAPExplainer:
    def __init__(self, model, tokenizer, device, model_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type
        self.preprocessor = Preprocessor()
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        print(f"Initializing SHAP KernelExplainer for {self.model_type}...")

        def predict_fn(texts: List[str]) -> np.ndarray:
            cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
            
            if hasattr(self.tokenizer, 'texts_to_sequences'):
                sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
            else:
                sequences = [[0]] * len(cleaned_texts)
            
            max_len = 150
            padded = []
            for seq in sequences:
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                else:
                    seq = seq[:max_len]
                padded.append(seq)
            
            inputs = torch.tensor(padded, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                output = self.model(inputs)
                if output.shape[-1] == 1:
                    probs = torch.sigmoid(output).cpu().numpy()
                else:
                    probs = torch.softmax(output, dim=1).cpu().numpy()
            
            return probs.flatten()

        background_texts = [
            "your account is secure",
            "click here to verify",
            "meeting at 3pm",
            "please review the document",
            "urgent action required",
            "password reset request",
            "delivery confirmation",
            "newsletter subscription",
            "thank you for your order",
            "best regards"
        ]
        
        self.explainer = shap.KernelExplainer(predict_fn, background_texts)
        print(f"SHAP explainer initialized for {self.model_type}")

    def explain(self, text: str, top_k: int = 10) -> List[Dict[str, float]]:
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = cleaned_text.split()
        
        def predict_single(text_input: List[str]) -> np.ndarray:
            return self.explainer.model(text_input)

        try:
            shap_values = self.explainer.shap_values([cleaned_text])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                phishing_idx = 1
                shap_values = shap_values[0, phishing_idx]
            else:
                shap_values = shap_values[0]
            
            token_scores = []
            for i, token in enumerate(tokens):
                if i < len(shap_values):
                    token_scores.append({"token": token, "shap_score": float(shap_values[i])})
                else:
                    token_scores.append({"token": token, "shap_score": 0.0})
            
            token_scores.sort(key=lambda x: abs(x["shap_score"]), reverse=True)
            return token_scores[:top_k]
            
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return [{"token": t, "shap_score": 0.0} for t in tokens[:top_k]]
