from fastapi import FastAPI

from interview_ai.api.routes.document import router as document_router
from interview_ai.api.routes.health import router as health_router
from interview_ai.api.routes.search import router as search_router
from interview_ai.api.routes.test import router as test_router
from interview_ai.config import config


def create_app() -> FastAPI:
    app = FastAPI(
        title="Interview AI",
        version="0.1.0",
        docs_url="/docs" if config.is_debug else None,
        redoc_url="/redoc" if config.is_debug else None,
    )
    app.include_router(health_router)
    app.include_router(document_router)
    app.include_router(search_router)
    app.include_router(test_router)
    return app


app = create_app()
