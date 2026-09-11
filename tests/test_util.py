from datetime import UTC, datetime, timedelta

import jwt
import pytest

from interview_ai.config import config
from interview_ai.util import JWT_ALGORITHM, jwt_encode, jwt_verify


@pytest.fixture(autouse=True)
def jwt_config(monkeypatch):
    monkeypatch.setattr(config, "jwt_secret", "test-only-secret-with-at-least-32-bytes")
    monkeypatch.setattr(config, "jwt_exp", 300)


def test_jwt_round_trip_includes_required_claims():
    payload = {"sub": "user-001", "role": "admin"}

    print(jwt_encode(payload))
    decoded = jwt_verify(jwt_encode(payload))

    assert decoded["sub"] == "user-001"
    assert decoded["role"] == "admin"
    assert isinstance(decoded["exp"], int)
    assert "exp" not in payload


@pytest.mark.parametrize("payload", [{}, {"sub": ""}, {"sub": 123}])
def test_jwt_encode_rejects_missing_or_invalid_subject(payload):
    with pytest.raises(ValueError, match="missing required claim: sub"):
        jwt_encode(payload)


def test_jwt_verify_requires_expiration_claim():
    token = jwt.encode({"sub": "user-001"}, config.jwt_secret, algorithm=JWT_ALGORITHM)

    with pytest.raises(jwt.MissingRequiredClaimError):
        jwt_verify(token)


def test_jwt_verify_rejects_expired_token():
    token = jwt.encode(
        {
            "sub": "user-001",
            "exp": datetime.now(UTC) - timedelta(seconds=1),
        },
        config.jwt_secret,
        algorithm=JWT_ALGORITHM,
    )

    with pytest.raises(jwt.ExpiredSignatureError):
        jwt_verify(token)


def test_jwt_functions_require_configured_secret(monkeypatch):
    monkeypatch.setattr(config, "jwt_secret", None)

    with pytest.raises(RuntimeError, match="JWT_SECRET is not configured"):
        jwt_encode({"sub": "user-001"})
