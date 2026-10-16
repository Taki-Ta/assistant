from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from zhiyin.api.routes.ai import router as ai_router
from zhiyin.api.routes.document import router as document_router
from zhiyin.api.routes.health import router as health_router
from zhiyin.api.routes.knowledge_base import router as knowledge_base_router
from zhiyin.api.routes.search import router as search_router
from zhiyin.api.routes.session import router as session_router
from zhiyin.api.routes.test import router as test_router
from zhiyin.config import config
from zhiyin.indexing.embeddings.openai import OpenAIEmbeddingProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncOpenAI(api_key=config.api_key, base_url=config.host) as client:
        app.state.embedding_provider = OpenAIEmbeddingProvider(
            client=client,
            model=config.embedding_model_name,
            dimensions=config.dimensions,
            batch_size=config.embedding_batch_size,
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Zhiyin",
        version="0.1.0",
        docs_url="/docs" if config.is_debug else None,
        redoc_url="/redoc" if config.is_debug else None,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(document_router)
    app.include_router(knowledge_base_router)
    app.include_router(search_router)
    app.include_router(test_router)
    app.include_router(ai_router)
    app.include_router(session_router)
    return app


app = create_app()
