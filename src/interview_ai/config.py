# import os

# from dotenv import load_dotenv

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = Field(alias="OPENAI_API_KEY")
    host: str = Field(alias="OPENAI_HOST")
    embedding_model_name: str = Field(alias="EMBEDDING_MODEL_NAME")

    dimensions: int = Field(default=256, alias="DIMENSIONS")
    is_debug: bool = Field(default=False, alias="IS_DEBUG")
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_exp: float = Field(default=300, alias="JWT_EXP_SECONDS")
    database_url: str = Field(alias="DATABASE_URL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# load_dotenv()


# class Config:
#     def __init__(self) -> None:
#         self.api_key = os.getenv("OPENAI_API_KEY")
#         self.host = os.getenv("OPENAI_HOST")
#         self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
#         self.dimensions = int(os.getenv("DIMENSIONS") or "256")
#         self.jwt_secret = os.getenv("JWT_SECRET")
#         self.jwt_exp = int(os.getenv("JWT_EXP_SECONDS") or "300")
#         self.is_debug = os.getenv("IS_DEBUG", "false").lower() == "true"


# config = Config()

config = Settings()
