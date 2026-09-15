# import os

# from dotenv import load_dotenv

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = Field(alias="OPENAI_API_KEY")
    host: str = Field(alias="OPENAI_HOST")
    embedding_model_name: str = Field(alias="EMBEDDING_MODEL_NAME")

    dimensions: int = Field(default=1536, alias="DIMENSIONS")
    embedding_batch_size: int = Field(default=20, ge=1, alias="EMBEDDING_BATCH_SIZE")
    search_top_k: int = Field(default=5, ge=1, alias="SEARCH_TOP_K")
    search_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="SEARCH_SCORE_THRESHOLD",
    )
    is_debug: bool = Field(default=False, alias="IS_DEBUG")
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_exp: float = Field(default=300, alias="JWT_EXP_SECONDS")
    database_url: str = Field(alias="DATABASE_URL")
    chat_model: str = Field(alias="CHAT_MODEL")
    chat_api_host: str = Field(alias="CHAT_API_HOST")
    chat_api_key: str = Field(alias="CHAT_API_KEY")
    max_tool_calls: int = Field(default=8, ge=1, alias="MAX_TOOL_CALLS")

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
