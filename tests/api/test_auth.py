from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from interview_ai.api.app import create_app
from interview_ai.config import config
from interview_ai.util import JWT_ALGORITHM, jwt_encode


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
