from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
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


class TurnStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemType(StrEnum):
    MESSAGE = "message"
    REASONING = "reasoning"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    COMPACTION = "compaction"


class ItemRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


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


class Session(SQLModel, table=True):
    """用户拥有的 AI 会话；不依赖具体模型供应商的会话对象。"""

    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_owner_updated", "owner_id", "updated_at"),)

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    owner_id: str = Field(sa_column=Column(String(128), nullable=False))
    title: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    is_deleted: bool = Field(
        default=False,
        sa_column=Column(
            Boolean,
            nullable=False,
            server_default=text("false"),
        ),
    )
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


class Turn(SQLModel, table=True):
    """从一次用户输入开始，到产生最终回答为止的完整执行过程。"""

    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "sequence",
            name="uq_turns_session_sequence",
        ),
        CheckConstraint("sequence >= 0", name="ck_turns_sequence"),
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'failed', 'cancelled')",
            name="ck_turns_status",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_turns_input_tokens",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_turns_output_tokens",
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_turns_total_tokens",
        ),
        Index("ix_turns_session_id", "session_id"),
        Index("ix_turns_status_updated", "status", "updated_at"),
    )

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    session_id: UUID = Field(
        sa_column=Column(
            ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    sequence: int = Field(sa_column=Column(Integer, nullable=False))
    status: TurnStatus = Field(
        default=TurnStatus.IN_PROGRESS,
        sa_column=Column(
            String(32),
            nullable=False,
            server_default=text("'in_progress'"),
        ),
    )
    provider: str | None = Field(
        default=None,
        sa_column=Column(String(64), nullable=True),
    )
    model: str | None = Field(
        default=None,
        sa_column=Column(String(128), nullable=True),
    )
    final_provider_response_id: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
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
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class ConversationItem(SQLModel, table=True):
    """按顺序保存消息、推理、工具调用、工具结果和压缩项。"""

    __tablename__ = "conversation_items"
    __table_args__ = (
        UniqueConstraint(
            "turn_id",
            "sequence",
            name="uq_conversation_items_turn_sequence",
        ),
        CheckConstraint(
            "sequence >= 0",
            name="ck_conversation_items_sequence",
        ),
        CheckConstraint(
            "role IS NULL OR role IN ('system', 'developer', 'user', 'assistant')",
            name="ck_conversation_items_role",
        ),
        Index("ix_conversation_items_turn_id", "turn_id"),
        Index("ix_conversation_items_call_id", "call_id"),
    )

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    turn_id: UUID = Field(
        sa_column=Column(
            ForeignKey("turns.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    sequence: int = Field(sa_column=Column(Integer, nullable=False))
    item_type: ItemType = Field(sa_column=Column(String(64), nullable=False))
    role: ItemRole | None = Field(
        default=None,
        sa_column=Column(String(32), nullable=True),
    )
    provider_item_id: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    call_id: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    tool_name: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    text_content: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    payload: dict[str, object] = Field(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=False,
            server_default=text("'{}'::jsonb"),
        ),
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True), nullable=False, server_default=func.now()
        ),
    )
