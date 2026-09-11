import hashlib
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from interview_ai.api.app import create_app
from interview_ai.api.auth import verify_bearer
from interview_ai.db.database import get_db
from interview_ai.db.models import Document


class _ScalarResult:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def all(self) -> list[Document]:
        return self._documents


class _DatabaseSession:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self._documents = documents or []
        self.added: Document | None = None

    async def scalars(self, statement: object) -> _ScalarResult:
        del statement
        return _ScalarResult(self._documents)

    def add(self, document: Document) -> None:
        self.added = document

    async def commit(self) -> None:
        return None

    async def refresh(self, document: Document) -> None:
        del document


def _client(session: _DatabaseSession) -> TestClient:
    async def override_get_db():
        yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_bearer] = lambda: {"sub": "user-001"}
    return TestClient(app)


def test_documents_endpoint_returns_markdown_from_postgres() -> None:
    now = datetime.now(UTC)
    content = "# Python"
    document = Document(
        owner_id="user-001",
        name="python.md",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        size_bytes=len(content.encode()),
        created_at=now,
        updated_at=now,
    )

    response = _client(_DatabaseSession([document])).get("/api/v1/documents")

    assert response.status_code == 200
    payload = response.json()[0]
    assert payload["id"] == str(document.id)
    assert payload["name"] == "python.md"
    assert payload["content"] == content
    assert "path" not in payload


def test_upload_markdown_saves_content_and_generates_uuid7() -> None:
    session = _DatabaseSession()
    content = "# Python\n\n基础内容。".encode()

    response = _client(session).post(
        "/api/v1/documents",
        files={"file": ("python.md", content, "text/markdown")},
    )

    assert response.status_code == 201
    assert session.added is not None
    assert session.added.id.version == 7
    assert session.added.owner_id == "user-001"
    assert session.added.name == "python.md"
    assert session.added.content == content.decode()
    assert session.added.size_bytes == len(content)
    assert session.added.content_hash == hashlib.sha256(content).hexdigest()


def test_upload_rejects_non_markdown_file() -> None:
    response = _client(_DatabaseSession()).post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"notes", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_non_utf8_markdown() -> None:
    response = _client(_DatabaseSession()).post(
        "/api/v1/documents",
        files={"file": ("notes.md", b"\xff\xfe", "text/markdown")},
    )

    assert response.status_code == 400
