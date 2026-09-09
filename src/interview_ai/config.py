import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    api_key = os.getenv("OPENAI_API_KEY")
    host = os.getenv("OPENAI_HOST")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    dimensions = int(os.getenv("DIMENSIONS", "256"))


config = Config()
