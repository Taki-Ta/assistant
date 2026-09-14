from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from interview_ai.indexing.models import SearchResult

from ..dependencies import Claims, SearchServiceDependency

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


def _owner_id(claims: dict[str, object]) -> str:
    owner_id = claims.get("sub")
    if not isinstance(owner_id, str) or not owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return owner_id


@router.post("", response_model=list[SearchResult])
async def search(
    request: SearchRequest,
    claims: Claims,
    search_service: SearchServiceDependency,
) -> list[SearchResult]:
    """根据用户问题查询知识库"""
    if not request.query.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="未传入要查询的问题")
    return await search_service.search(request.query, _owner_id(claims), request.limit)
