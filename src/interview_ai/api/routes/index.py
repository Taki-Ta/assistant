from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.api.auth import verify_bearer
from interview_ai.config import config
from interview_ai.db.database import get_db
from interview_ai.db.models import Document, DocumentStatus
from interview_ai.db.repositories import PostgresChunkStore
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider
from interview_ai.indexing.service import IndexService
from interview_ai.indexing.vector_store.pg_vector_store import PGVectorStore
from interview_ai.ingestion.scanner import split_document

router = APIRouter(prefix="/api/v1/index", tags=["index"])
Claims = Annotated[dict[str, object], Depends(verify_bearer)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


def _owner_id(claims: dict[str, object]) -> str:
    owner_id = claims.get("sub")
    if not isinstance(owner_id, str) or not owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return owner_id


def get_index_service(db: DatabaseSession) -> IndexService:
    client = OpenAI(api_key=config.api_key, base_url=config.host)
    embedding_provider = OpenAIEmbeddingProvider(
        client=client,
        model=config.embedding_model_name,
        dimensions=config.dimensions,
    )
    return IndexService(
        embedding_provider=embedding_provider,
        chunk_store=PostgresChunkStore(db),
        vector_store=PGVectorStore(db),
    )


IndexServiceDependency = Annotated[IndexService, Depends(get_index_service)]


@router.post("", response_model=Document)
async def index_document(
    document_id: str,
    claims: Claims,
    db: DatabaseSession,
    index_service: IndexServiceDependency,
) -> Document:
    """切分一个当前用户的文档，并将 Chunk 和向量写入 PostgreSQL。"""
    try:
        parsed_document_id = UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id 应为 UUID",
        ) from exc

    statement = select(Document).where(
        Document.owner_id == _owner_id(claims),
        Document.id == parsed_document_id,
    )
    document = (await db.scalars(statement)).one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到文档",
        )

    document.status = DocumentStatus.INDEXING
    document.error_message = None
    chunks = split_document(document)
    document.chunk_count = await index_service.index(document, chunks)
    document.status = DocumentStatus.INDEXED
    document.indexed_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(document)
    return document
