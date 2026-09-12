import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

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


def is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个非零、同维向量的余弦相似度。"""
    if len(a) != len(b):
        raise ValueError("向量长度不一致，无法计算余弦相似度")

    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量无法计算余弦相似度")
    return dot_product / (norm_a * norm_b)
