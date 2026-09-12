import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.api.auth import verify_bearer
from interview_ai.config import config
from interview_ai.db.database import get_db
from interview_ai.db.models import Document, DocumentStatus
from interview_ai.db.repositories import PostgresChunkStore
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider
from interview_ai.indexing.service import IndexService
from interview_ai.indexing.vector_store import PostgresVectorStore
from interview_ai.ingestion.scanner import split_document

MAX_MARKDOWN_SIZE = 5 * 1024 * 1024

router = APIRouter(prefix="/api/v1/documents", tags=["document"])
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
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.host)
    embedding_provider = OpenAIEmbeddingProvider(
        client=client,
        model=config.embedding_model_name,
        dimensions=config.dimensions,
    )
    return IndexService(
        embedding_provider=embedding_provider,
        chunk_store=PostgresChunkStore(db),
        vector_store=PostgresVectorStore(db),
    )


IndexServiceDependency = Annotated[IndexService, Depends(get_index_service)]


@router.get("", response_model=list[Document], status_code=status.HTTP_200_OK)
async def get_documents(claims: Claims, db: DatabaseSession) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.owner_id == _owner_id(claims))
        .order_by(Document.updated_at.desc())
    )
    result = await db.scalars(stmt)
    return list(result.all())


@router.post("", response_model=Document, status_code=status.HTTP_201_CREATED)
async def add_document(
    file: Annotated[UploadFile, File(description="UTF-8 Markdown 文件")],
    claims: Claims,
    db: DatabaseSession,
) -> Document:
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".md":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="只支持 .md 文件",
        )

    content_bytes = await file.read(MAX_MARKDOWN_SIZE + 1)
    if len(content_bytes) > MAX_MARKDOWN_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Markdown 文件不能超过 5 MiB",
        )

    try:
        content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Markdown 文件必须使用 UTF-8 编码",
        ) from exc

    document = Document(
        owner_id=_owner_id(claims),
        name=filename,
        content=content,
        content_hash=hashlib.sha256(content_bytes).hexdigest(),
        size_bytes=len(content_bytes),
        mime_type=file.content_type or "text/markdown",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


@router.post("/{document_id}/index", response_model=Document)
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
