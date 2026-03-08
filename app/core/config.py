from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Phishing Email Detection API"

    # MODEL_PATH_BERT: str  # [BERT_RESTORE] Uncomment to enable BERT model
    MODEL_PATH_LSTM: str
    MODEL_PATH_GRU: str
    TOKENIZER_PATH_RNN: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
