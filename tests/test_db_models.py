from interview_ai.db.models import Document


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
