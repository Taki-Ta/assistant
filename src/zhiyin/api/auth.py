from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from zhiyin.util import jwt_verify

bearer_schema = HTTPBearer(auto_error=False)


def verify_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_schema)],
) -> dict[str, object]:
    if credentials is None:
        raise _unauthorized("Missing bearer token")

    try:
        return jwt_verify(credentials.credentials)
    except jwt.PyJWTError as error:
        raise _unauthorized("Invalid or expired token") from error


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


Claims = Annotated[dict[str, object], Depends(verify_bearer)]


def get_current_user_id(claims: Claims) -> str:
    owner_id = claims.get("sub")
    if not isinstance(owner_id, str) or not owner_id:
        raise _unauthorized("Token subject is missing")
    return owner_id


CurrentUserId = Annotated[str, Depends(get_current_user_id)]
