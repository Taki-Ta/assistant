"""Knowledge-base permissions shared by document, vector and citation queries."""

from hashlib import md5
from typing import Literal
from uuid import UUID

from sqlalchemy import exists, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Document, KnowledgeBase, KnowledgeBaseMember

AccessAction = Literal["read", "edit", "manage", "own"]
ROLES = {
    "read": ("viewer", "editor", "admin"),
    "edit": ("editor", "admin"),
    "manage": ("admin",),
    "own": (),
}


def accessible_knowledge_bases(user_id: str, action: AccessAction = "read"):
    membership = exists().where(
        KnowledgeBaseMember.knowledge_base_id == KnowledgeBase.id,
        KnowledgeBaseMember.user_id == user_id,
        KnowledgeBaseMember.role.in_(ROLES[action]),
    )
    return select(KnowledgeBase.id).where(
        KnowledgeBase.is_deleted.is_(False),
        or_(KnowledgeBase.owner_id == user_id, membership),
    )


def document_access(user_id: str, action: AccessAction = "read"):
    return Document.knowledge_base_id.in_(accessible_knowledge_bases(user_id, action))


class KnowledgeBaseService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def require(
        self, user_id: str, knowledge_base_id: UUID, action: AccessAction = "read"
    ) -> KnowledgeBase:
        item = await self.db.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.id.in_(accessible_knowledge_bases(user_id, action)),
            )
        )
        if item is None:
            raise LookupError("知识库不存在或无权访问")
        return item

    async def default_id(self, user_id: str) -> UUID:
        # Same deterministic ID as the migration; concurrent uploads are safe.
        identifier = UUID(
            hex=md5(("default-knowledge-base:" + user_id).encode()).hexdigest()
        )
        statement = (
            insert(KnowledgeBase)
            .values(
                id=identifier,
                owner_id=user_id,
                name="默认知识库",
                is_default=True,
            )
            .on_conflict_do_nothing()
        )
        await self.db.execute(statement)
        await self.require(user_id, identifier, "edit")
        return identifier
