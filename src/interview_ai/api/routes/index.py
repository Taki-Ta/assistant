from interview_ai.api.routers import protected_router


@protected_router.get("/test", tags=["index"])
def test_authentication() -> dict[str, str]:
    return {"message": "authenticated"}
