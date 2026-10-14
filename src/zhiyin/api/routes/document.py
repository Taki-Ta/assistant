import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from zhiyin.db.knowledge_bases import KnowledgeBaseService, document_access
from zhiyin.db.models import Chunk, Document, DocumentStatus
from zhiyin.ingestion.scanner import split_document

from ..dependencies import Claims, DatabaseSession, IndexServiceDependency

MAX_MARKDOWN_SIZE = 5 * 1024 * 1024

router = APIRouter(prefix="/api/v1/documents", tags=["document"])


def _owner_id(claims: dict[str, object]) -> str:
    owner_id = claims.get("sub")
    if not isinstance(owner_id, str) or not owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return owner_id


@router.get("", response_model=list[Document], status_code=status.HTTP_200_OK)
async def get_documents(
    claims: Claims, db: DatabaseSession, knowledge_base_id: UUID | None = None
) -> list[Document]:
    stmt = (
        select(Document)
        .where(document_access(_owner_id(claims)))
        .order_by(Document.updated_at.desc())
    )
    if knowledge_base_id is not None:
        stmt = stmt.where(Document.knowledge_base_id == knowledge_base_id)
    items = (await db.scalars(stmt)).all()
    return [
        item.model_copy(
            update={
                "content": item.content[:30] + "..."
                if len(item.content) > 30
                else item.content
            }
        )
        for item in items
    ]


@router.post("", response_model=Document, status_code=status.HTTP_201_CREATED)
async def add_document(
    file: Annotated[UploadFile, File(description="UTF-8 Markdown 文件")],
    claims: Claims,
    db: DatabaseSession,
    knowledge_base_id: Annotated[UUID | None, Form()] = None,
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

    owner_id = _owner_id(claims)
    knowledge_bases = KnowledgeBaseService(db)
    try:
        if knowledge_base_id is None:
            knowledge_base_id = await knowledge_bases.default_id(owner_id)
        else:
            await knowledge_bases.require(owner_id, knowledge_base_id, "edit")
    except LookupError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    document = Document(
        owner_id=owner_id,
        knowledge_base_id=knowledge_base_id,
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
    """切分有编辑权限的文档，并将 Chunk 和向量写入 PostgreSQL。"""
    try:
        parsed_document_id = UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id 应为 UUID",
        ) from exc

    statement = select(Document).where(
        document_access(_owner_id(claims), "edit"),
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


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: UUID, claims: Claims, db: DatabaseSession
) -> Document:
    item = await db.scalar(
        select(Document).where(
            Document.id == document_id, document_access(_owner_id(claims))
        )
    )
    if item is None:
        raise HTTPException(404, "未找到文档")
    return item


@router.get("/{document_id}/chunks", response_model=list[Chunk])
async def get_document_chunks(
    document_id: UUID, claims: Claims, db: DatabaseSession
) -> list[Chunk]:
    await get_document(document_id, claims, db)
    result = await db.scalars(
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.document_id == document_id, document_access(_owner_id(claims)))
        .order_by(Chunk.sort_index)
    )
    return list(result)
