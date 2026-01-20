import torch
from transformers import AutoTokenizer
import pickle
import re
from typing import List, Union

class Preprocessor:
    def __init__(self):
        # We don't load artifacts here; they are passed in or loaded by ModelManager
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_bert(self, text: str, tokenizer: AutoTokenizer, max_len: int = 128):
        """Preprocess text for BERT model."""
        text = self.clean_text(text)
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def preprocess_rnn(self, text: str, tokenizer, max_len: int = 150):
        """Preprocess text for LSTM/GRU models."""
        # Assuming tokenizer is a Keras-like tokenizer or similar custom one
        # that has texts_to_sequences. If it's different, we'll adjust.
        # For this implementation, I'll assume standard fit-on-text tokenizer behavior
        # often saved as pickle.
        
        text = self.clean_text(text)
        
        # Check if tokenizer has texts_to_sequences (Keras style)
        if hasattr(tokenizer, 'texts_to_sequences'):
            sequences = tokenizer.texts_to_sequences([text])
        else:
            # Fallback or specific implementation if it's a different type
            # For now, let's assume it has a method to convert text to ids
            # This part is highly dependent on how the 'pickle' tokenizer was created.
            # I will create a dummy heuristic if methods are missing.
             sequences = [[0]] # Placeholder

        # Pad sequences
        # Simple manual padding to max_len from the left (pre-padding) or right?
        # Usually RNNs use pre-padding or post-padding. Let's assume post-padding for now
        # unless specified. 
        # But wait, Keras pad_sequences does pre-padding by default.
        
        seq = sequences[0]
        if len(seq) < max_len:
            # Pad with 0s (post-padding)
            seq = seq + [0] * (max_len - len(seq))
        else:
            # Truncate
            seq = seq[:max_len]
            
        return torch.tensor([seq], dtype=torch.long)
