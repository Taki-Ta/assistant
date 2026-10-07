# import os

# from dotenv import load_dotenv

from typing import Self

from pydantic import Field, model_validator
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
    chat_context_window: int = Field(
        default=32768,
        gt=0,
        alias="CHAT_CONTEXT_WINDOW",
        description="模型允许的最大上下文 Token 数，请按模型服务说明设置",
    )
    chat_context_budget: int = Field(
        default=32768,
        gt=0,
        alias="CHAT_CONTEXT_BUDGET",
        description="单次模型请求的总 Token 预算，包含输入和输出预留",
    )
    max_tool_calls: int = Field(default=8, ge=1, alias="MAX_TOOL_CALLS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @model_validator(mode="after")
    def validate_chat_context_budget(self) -> Self:
        if self.chat_context_budget > self.chat_context_window:
            raise ValueError(
                "CHAT_CONTEXT_BUDGET 不能超过 CHAT_CONTEXT_WINDOW；"
                "请降低请求预算，或按模型服务说明设置上下文容量"
            )
        return self


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
