import hashlib
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.api.auth import verify_bearer
from interview_ai.db.database import get_db
from interview_ai.db.models import Document

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
    print(f'{file=}')
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
