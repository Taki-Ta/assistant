from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zhiyin.api.app import create_app
from zhiyin.api.auth import CurrentUserId, verify_bearer
from zhiyin.api.dependencies import AgentContextDependency, get_search_service
from zhiyin.config import config
from zhiyin.util import JWT_ALGORITHM, jwt_encode


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "jwt_secret", "test-only-secret-with-at-least-32-bytes")
    monkeypatch.setattr(config, "jwt_exp", 300)
    return TestClient(create_app())


def test_protected_endpoint_accepts_valid_bearer_token(client):
    token = jwt_encode({"sub": "user-001"})

    response = client.get("/api/v1/test", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"message": "authenticated"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer invalid-token"},
        {"Authorization": "Basic credentials"},
    ],
)
def test_protected_endpoint_rejects_missing_or_invalid_credentials(client, headers):
    response = client.get("/api/v1/test", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_protected_endpoint_rejects_expired_token(client):
    token = jwt.encode(
        {
            "sub": "user-001",
            "exp": datetime.now(UTC) - timedelta(seconds=1),
        },
        config.jwt_secret,
        algorithm=JWT_ALGORITHM,
    )

    response = client.get("/api/v1/test", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}


def test_identity_dependency_and_agent_context_use_the_same_user():
    app = FastAPI()
    app.dependency_overrides[verify_bearer] = lambda: {"sub": "user-001"}

    @app.get("/identity")
    def identity(owner_id: CurrentUserId, context: AgentContextDependency):
        return {"owner_id": owner_id, "context_owner_id": context.owner_id}

    response = TestClient(app).get("/identity")

    assert response.status_code == 200
    assert response.json() == {
        "owner_id": "user-001",
        "context_owner_id": "user-001",
    }


@pytest.mark.parametrize("claims", [{}, {"sub": ""}, {"sub": None}, {"sub": 123}])
@pytest.mark.parametrize(
    "method, path, body",
    [
        ("GET", "/api/v1/documents", None),
        ("GET", "/api/v1/knowledge-bases", None),
        ("POST", "/api/v1/search", {"query": "Python"}),
    ],
)
def test_routes_reject_invalid_user_identity(claims, method, path, body):
    app = create_app()
    search_service = AsyncMock()
    app.dependency_overrides[verify_bearer] = lambda: claims
    app.dependency_overrides[get_search_service] = lambda: search_service

    response = TestClient(app).request(method, path, json=body)

    assert response.status_code == 401
    assert response.json() == {"detail": "Token subject is missing"}
    assert response.headers["www-authenticate"] == "Bearer"
    search_service.search.assert_not_awaited()
