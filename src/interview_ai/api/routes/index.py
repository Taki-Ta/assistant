from fastapi import APIRouter, Depends

from interview_ai.api.auth import verify_bearer

protected_router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_bearer)],
)


@protected_router.get("/test", tags=["index"])
def test_authentication() -> dict[str, str]:
    return {"message": "authenticated"}
