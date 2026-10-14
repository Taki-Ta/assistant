from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from zhiyin.db.knowledge_bases import accessible_knowledge_bases, document_access
from zhiyin.db.models import Document, KnowledgeBase, KnowledgeBaseMember


@pytest.fixture
def database():
    engine = create_engine("sqlite://")
    for model in (KnowledgeBase, KnowledgeBaseMember, Document):
        model.__table__.create(engine)
    with Session(engine) as db:
        db.add_all(
            [
                KnowledgeBase(id=UUID(int=1), owner_id="owner", name="shared"),
                KnowledgeBase(id=UUID(int=2), owner_id="other", name="private"),
                KnowledgeBase(
                    id=UUID(int=3), owner_id="owner", name="deleted", is_deleted=True
                ),
            ]
        )
        for user, role in [
            ("reader", "viewer"),
            ("writer", "editor"),
            ("manager", "admin"),
        ]:
            db.add(
                KnowledgeBaseMember(
                    knowledge_base_id=UUID(int=1), user_id=user, role=role
                )
            )
        for index, uploader in [(1, "writer"), (2, "owner"), (3, "owner")]:
            db.add(
                Document(
                    id=UUID(int=index),
                    knowledge_base_id=UUID(int=index),
                    owner_id=uploader,
                    name="doc",
                    content="text",
                    content_hash="hash",
                    size_bytes=4,
                )
            )
        db.commit()
        yield db
    engine.dispose()


@pytest.mark.parametrize(
    "user,actions",
    [
        ("owner", {"read", "edit", "manage", "own"}),
        ("reader", {"read"}),
        ("writer", {"read", "edit"}),
        ("manager", {"read", "edit", "manage"}),
        ("outsider", set()),
    ],
)
def test_roles_are_enforced_in_database_queries(database, user, actions):
    for action in ("read", "edit", "manage", "own"):
        ids = list(database.scalars(accessible_knowledge_bases(user, action)))
        assert ids == ([UUID(int=1)] if action in actions else [])


def test_document_access_follows_library_not_uploader_and_revokes_immediately(database):
    query = select(Document.id).where(document_access("reader"))
    assert list(database.scalars(query)) == [UUID(int=1)]
    membership = database.get(KnowledgeBaseMember, (UUID(int=1), "reader"))
    database.delete(membership)
    database.commit()
    assert list(database.scalars(query)) == []
    # Uploading a document grants no access to a library owned by somebody else.
    assert list(
        database.scalars(select(Document.id).where(document_access("owner")))
    ) == [UUID(int=1)]


def test_deleted_knowledge_base_is_unreadable_even_by_owner(database):
    assert (
        UUID(int=3) not in database.scalars(accessible_knowledge_bases("owner")).all()
    )
