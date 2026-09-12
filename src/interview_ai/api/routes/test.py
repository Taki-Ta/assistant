from fastapi import APIRouter, Depends

from interview_ai.api.auth import verify_bearer

router = APIRouter(prefix="/api/v1/test", tags=["test"])


@router.get("/", dependencies=[Depends(verify_bearer)])
def test_authentication() -> dict[str, str]:
    return {"message": "authenticated"}
