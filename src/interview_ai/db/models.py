from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
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
        sa_column=Column(String(32), nullable=False, server_default=text("'uploaded'")),
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


class Chunk(SQLModel, table=True):
    """Markdown 切块；向量本身由向量数据库保存。"""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "sort_index",
            name="uq_chunks_document_sort_index",
        ),
        CheckConstraint("sort_index >= 0", name="ck_chunks_sort_index"),
        CheckConstraint("level BETWEEN 0 AND 6", name="ck_chunks_level"),
        CheckConstraint("start_line >= 1", name="ck_chunks_start_line"),
        CheckConstraint("end_line >= start_line", name="ck_chunks_line_range"),
        Index("ix_chunks_document_id", "document_id"),
    )

    id: UUID = Field(primary_key=True)
    document_id: UUID = Field(
        sa_column=Column(
            ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    sort_index: int = Field(sa_column=Column(Integer, nullable=False))
    level: int = Field(sa_column=Column(Integer, nullable=False))
    headings: tuple[str, ...] = Field(
        default_factory=tuple,
        sa_column=Column(ARRAY(Text), nullable=False),
    )
    content: str = Field(sa_column=Column(Text, nullable=False))
    start_line: int = Field(sa_column=Column(Integer, nullable=False))
    end_line: int = Field(sa_column=Column(Integer, nullable=False))
    hash: str = Field(sa_column=Column("content_hash", String(64), nullable=False))
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True), nullable=False, server_default=func.now()
        ),
    )


class VectorRecord(SQLModel, table=True):
    """Chunk 对应的向量；正文及结构化元数据保留在 chunks 表。"""

    __tablename__ = "vectors"
    __table_args__ = (
        Index(
            "ix_vectors_vector_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )

    chunk_id: UUID = Field(
        sa_column=Column(
            ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    vector: list[float] = Field(
        sa_column=Column(VECTOR(1536), nullable=False),
    )
    embedding_model: str = Field(sa_column=Column(String(128), nullable=False))
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
