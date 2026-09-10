from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from interview_ai.config import config

JWT_ALGORITHM = "HS256"


def _jwt_secret() -> str:
    if not config.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured")
    return config.jwt_secret


def jwt_verify(token: str) -> dict[str, Any]:
    return jwt.decode(
        token,
        _jwt_secret(),
        algorithms=[JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )


def jwt_encode(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload.get("sub"), str) or not payload["sub"]:
        raise ValueError("missing required claim: sub")
    claims = dict(payload)

    claims["exp"] = datetime.now(UTC) + timedelta(seconds=config.jwt_exp)
    return jwt.encode(claims, _jwt_secret(), algorithm=JWT_ALGORITHM)
