from interview_ai.api.routers import public_router


@public_router.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "hello world"}


@public_router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
