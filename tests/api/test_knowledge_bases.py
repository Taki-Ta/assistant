from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.api.app import create_app
from interview_ai.api.auth import verify_bearer
from interview_ai.db.database import get_db
from interview_ai.db.models import KnowledgeBase


def client(db, user="owner"):
    async def override():
        yield db

    app = create_app()
    app.dependency_overrides[get_db] = override
    app.dependency_overrides[verify_bearer] = lambda: {"sub": user}
    return TestClient(app)


def database():
    db = MagicMock(spec=AsyncSession)
    db.scalar = AsyncMock(
        return_value=KnowledgeBase(id=UUID(int=1), owner_id="owner", name="library")
    )
    db.scalars = AsyncMock(return_value=[])
    return db


def test_create_library_uses_authenticated_owner():
    db = database()
    response = client(db).post("/api/v1/knowledge-bases", json={"name": "Python"})
    assert response.status_code == 201
    assert db.add.call_args.args[0].owner_id == "owner"
    assert response.json()["is_default"] is False
    assert (
        client(db)
        .post("/api/v1/knowledge-bases", json={"name": " ", "owner_id": "other"})
        .status_code
        == 422
    )


@pytest.mark.parametrize(
    "method,path,data",
    [
        ("get", "/api/v1/knowledge-bases/00000000-0000-0000-0000-000000000001", None),
        (
            "put",
            "/api/v1/knowledge-bases/00000000-0000-0000-0000-000000000001/members/reader",
            {"role": "viewer"},
        ),
        ("post", "/api/v1/documents", None),
    ],
)
def test_inaccessible_library_never_returns_or_writes_content(method, path, data):
    db = database()
    db.scalar.return_value = None
    kwargs = {"json": data} if data else {}
    if path == "/api/v1/documents":
        kwargs = {
            "files": {"file": ("doc.md", b"text", "text/markdown")},
            "data": {"knowledge_base_id": str(UUID(int=1))},
        }
    assert client(db).request(method, path, **kwargs).status_code == 404
    db.execute.assert_not_awaited()
    db.commit.assert_not_awaited()


def test_owner_cannot_be_modified_as_member_and_default_cannot_be_deleted():
    db = database()
    root = f"/api/v1/knowledge-bases/{UUID(int=1)}"
    assert (
        client(db).put(root + "/members/owner", json={"role": "viewer"}).status_code
        == 422
    )
    assert (
        client(db).put(root + "/members/reader", json={"role": "owner"}).status_code
        == 422
    )
    db.scalar.return_value.is_default = True
    assert client(db).delete(root).status_code == 409
    db.commit.assert_not_awaited()


def test_member_upsert_uses_target_library_and_commits():
    db = database()
    assert (
        client(db)
        .put(
            f"/api/v1/knowledge-bases/{UUID(int=1)}/members/reader",
            json={"role": "viewer"},
        )
        .status_code
        == 204
    )
    params = db.execute.await_args.args[0].compile().params
    assert params["knowledge_base_id"] == UUID(int=1)
    assert params["user_id"] == "reader"
    db.commit.assert_awaited_once()
