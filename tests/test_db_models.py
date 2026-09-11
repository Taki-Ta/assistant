from uuid import NAMESPACE_URL, uuid5

from interview_ai.db.models import Chunk, Document, VectorRecord


def test_document_id_is_generated_as_uuid7() -> None:
    document = Document(
        owner_id="user-001",
        name="python.md",
        content="# Python",
        content_hash="a" * 64,
        size_bytes=8,
    )

    assert document.id.version == 7


def test_postgres_document_table_stores_markdown_but_not_local_path() -> None:
    assert set(Document.__table__.columns.keys()) == {
        "id",
        "owner_id",
        "name",
        "content",
        "content_hash",
        "size_bytes",
        "mime_type",
        "status",
        "error_message",
        "chunk_count",
        "created_at",
        "updated_at",
        "indexed_at",
    }
    assert "path" not in Document.__table__.columns


def test_postgres_chunk_table_stores_content_but_not_vector() -> None:
    assert set(Chunk.__table__.columns.keys()) == {
        "id",
        "document_id",
        "sort_index",
        "level",
        "headings",
        "content",
        "start_line",
        "end_line",
        "content_hash",
        "created_at",
    }
    assert "vector" not in Chunk.__table__.columns


def test_chunk_accepts_deterministic_uuid() -> None:
    document_id = uuid5(NAMESPACE_URL, "document")
    chunk_id = uuid5(NAMESPACE_URL, f"{document_id}-0-{'a' * 64}")

    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        sort_index=0,
        level=1,
        headings=("Python",),
        content="# Python",
        start_line=1,
        end_line=1,
        hash="a" * 64,
    )

    assert chunk.id == chunk_id


def test_postgres_vector_table_only_stores_vector_data() -> None:
    assert set(VectorRecord.__table__.columns.keys()) == {
        "chunk_id",
        "vector",
        "embedding_model",
        "created_at",
        "updated_at",
    }
    assert "content" not in VectorRecord.__table__.columns
    assert "document_id" not in VectorRecord.__table__.columns
