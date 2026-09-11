from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"message": "hello world"}


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
