from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlmodel import Field, SQLModel
from uuid6 import uuid7


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(SQLModel, table=True):
    """保存在 PostgreSQL 中的 Markdown 文档及其索引状态。"""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("size_bytes >= 0", name="ck_documents_size"),
        CheckConstraint(
            "chunk_count IS NULL OR chunk_count >= 0",
            name="ck_documents_chunk_count",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'parsing', 'indexing', 'indexed', 'failed')",
            name="ck_documents_status",
        ),
        Index("ix_documents_owner_updated", "owner_id", "updated_at"),
        Index("ix_documents_status_updated", "status", "updated_at"),
    )

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    owner_id: str = Field(sa_column=Column(String(128), nullable=False))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    content: str = Field(sa_column=Column(Text, nullable=False))
    content_hash: str = Field(sa_column=Column(String(64), nullable=False))
    size_bytes: int = Field(sa_column=Column(BigInteger, nullable=False))
    mime_type: str = Field(
        default="text/markdown",
        sa_column=Column(
            String(255), nullable=False, server_default=text("'text/markdown'")
        ),
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED,
        sa_column=Column(
            String(32), nullable=False, server_default=text("'uploaded'")
        ),
    )
    error_message: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    chunk_count: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True), nullable=False, server_default=func.now()
        ),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )
    indexed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
