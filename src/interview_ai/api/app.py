from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from interview_ai.api.routes.ai import router as ai_router
from interview_ai.api.routes.document import router as document_router
from interview_ai.api.routes.health import router as health_router
from interview_ai.api.routes.search import router as search_router
from interview_ai.api.routes.test import router as test_router
from interview_ai.config import config
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncOpenAI(api_key=config.api_key, base_url=config.host) as client:
        app.state.embedding_provider = OpenAIEmbeddingProvider(
            client=client,
            model=config.embedding_model_name,
            dimensions=config.dimensions,
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Interview AI",
        version="0.1.0",
        docs_url="/docs" if config.is_debug else None,
        redoc_url="/redoc" if config.is_debug else None,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(document_router)
    app.include_router(search_router)
    app.include_router(test_router)
    app.include_router(ai_router)
    return app


app = create_app()
