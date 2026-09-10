from fastapi import FastAPI

from interview_ai.api.routes.health import public_router
from interview_ai.api.routes.index import protected_router
from interview_ai.config import config


def create_app() -> FastAPI:
    app = FastAPI(
        title="Interview AI",
        version="0.1.0",
        docs_url="/docs" if config.is_debug else None,
        redoc_url="/redoc" if config.is_debug else None,
    )
    app.include_router(public_router)
    app.include_router(protected_router)
    return app


app = create_app()
