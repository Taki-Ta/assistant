import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from interview_ai.api.app import create_app
from interview_ai.api.auth import verify_bearer
from interview_ai.api.routes.document import get_index_service
from interview_ai.db.database import get_db
from interview_ai.db.models import Document, DocumentStatus


class _ScalarResult:
    def __init__(self, document: Document | None) -> None:
        self._document = document

    def one_or_none(self) -> Document | None:
        return self._document


class _DatabaseSession:
    def __init__(self, document: Document | None) -> None:
        self._document = document
        self.commit = AsyncMock()
        self.refresh = AsyncMock()

    async def scalars(self, statement: object) -> _ScalarResult:
        del statement
        return _ScalarResult(self._document)


def _document() -> Document:
    content = "# Python\n\n基础内容。"
    now = datetime.now(UTC)
    return Document(
        owner_id="user-001",
        name="python.md",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        size_bytes=len(content.encode()),
        created_at=now,
        updated_at=now,
    )


def _client(session: _DatabaseSession, index_service: AsyncMock) -> TestClient:
    async def override_get_db():
        yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_bearer] = lambda: {"sub": "user-001"}
    app.dependency_overrides[get_index_service] = lambda: index_service
    return TestClient(app)


def test_index_document_persists_chunks_and_vectors() -> None:
    document = _document()
    session = _DatabaseSession(document)
    index_service = AsyncMock()
    index_service.index.return_value = 1

    response = _client(session, index_service).post(
        f"/api/v1/documents/{document.id!s}/index"
    )

    assert response.status_code == 200
    assert document.status == DocumentStatus.INDEXED
    assert document.chunk_count == 1
    assert document.indexed_at is not None
    indexed_chunks = index_service.index.await_args.args[1]
    assert indexed_chunks[0].document_id == document.id
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(document)


def test_index_document_rejects_invalid_uuid() -> None:
    session = _DatabaseSession(None)
    response = _client(session, AsyncMock()).post(
        "/api/v1/documents/not-a-uuid/index",
    )

    assert response.status_code == 400


def test_index_document_returns_not_found() -> None:
    session = _DatabaseSession(None)
    response = _client(session, AsyncMock()).post(
        "/api/v1/documents/550e8400-e29b-41d4-a716-446655440001/index",
    )

    assert response.status_code == 404
