from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from zhiyin.indexing.models import SearchResult

from ..auth import CurrentUserId
from ..dependencies import SearchServiceDependency

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)


@router.post("", response_model=list[SearchResult])
async def search(
    request: SearchRequest,
    owner_id: CurrentUserId,
    search_service: SearchServiceDependency,
) -> list[SearchResult]:
    """根据用户问题查询知识库"""
    if not request.query.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="未传入要查询的问题")
    return await search_service.search(request.query, owner_id, request.limit)
