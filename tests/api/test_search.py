from unittest.mock import AsyncMock
from uuid import UUID

from fastapi.testclient import TestClient

from interview_ai.api.app import create_app
from interview_ai.api.auth import verify_bearer
from interview_ai.api.routes.search import get_search_service
from interview_ai.indexing.models import SearchResult
from interview_ai.indexing.search_service import SearchService


def _service() -> AsyncMock:
    service = AsyncMock(spec=SearchService)
    service.search.return_value = [
        SearchResult(
            chunk_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            content="# Python",
            score=0.91,
        )
    ]
    return service


def _client(service: AsyncMock, *, authenticated: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_search_service] = lambda: service
    if authenticated:
        app.dependency_overrides[verify_bearer] = lambda: {"sub": "user-001"}
    return TestClient(app)


def test_search_returns_current_users_similar_chunks() -> None:
    service = _service()

    response = _client(service).post(
        "/api/v1/search",
        json={"query": "Python 是什么？", "limit": 3},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
            "content": "# Python",
            "score": 0.91,
        }
    ]
    service.search.assert_awaited_once_with("Python 是什么？", "user-001", 3)


def test_search_rejects_blank_query() -> None:
    service = _service()

    response = _client(service).post(
        "/api/v1/search",
        json={"query": "   ", "limit": 5},
    )

    assert response.status_code == 400
    service.search.assert_not_awaited()


def test_search_validates_limit() -> None:
    service = _service()

    response = _client(service).post(
        "/api/v1/search",
        json={"query": "Python", "limit": 0},
    )

    assert response.status_code == 422
    service.search.assert_not_awaited()


def test_search_requires_authentication() -> None:
    response = _client(_service(), authenticated=False).post(
        "/api/v1/search",
        json={"query": "Python", "limit": 5},
    )

    assert response.status_code == 401
