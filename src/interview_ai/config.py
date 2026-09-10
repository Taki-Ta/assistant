import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.host = os.getenv("OPENAI_HOST")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        self.dimensions = int(os.getenv("DIMENSIONS") or "256")
        self.jwt_secret = os.getenv("JWT_SECRET")
        self.jwt_exp = int(os.getenv("JWT_EXP_SECONDS") or "300")
        self.is_debug = os.getenv("IS_DEBUG", "false").lower() == "true"


config = Config()
