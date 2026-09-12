from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.api.auth import verify_bearer
from interview_ai.config import config
from interview_ai.db.database import get_db
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider
from interview_ai.indexing.models import SearchResult
from interview_ai.indexing.search_service import SearchService
from interview_ai.indexing.vector_store import PostgresVectorStore

router = APIRouter(prefix="/api/v1/search", tags=["search"])
Claims = Annotated[dict[str, object], Depends(verify_bearer)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


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


def get_search_service(db: DatabaseSession) -> SearchService:
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.host)
    embedding_provider = OpenAIEmbeddingProvider(
        client=client,
        model=config.embedding_model_name,
        dimensions=config.dimensions,
    )
    return SearchService(
        embedding_provider=embedding_provider,
        vector_store=PostgresVectorStore(db),
    )


SearchServiceDependency = Annotated[SearchService, Depends(get_search_service)]


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
