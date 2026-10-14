from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from interview_ai.db.knowledge_bases import (
    KnowledgeBaseService,
    accessible_knowledge_bases,
)
from interview_ai.db.models import KnowledgeBase, KnowledgeBaseMember, utc_now

from ..dependencies import Claims, DatabaseSession
from .document import _owner_id

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-base"])


class KnowledgeBaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)


class MemberInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["viewer", "editor", "admin"]


async def require(db, claims, identifier, action="read"):
    try:
        return await KnowledgeBaseService(db).require(
            _owner_id(claims), identifier, action
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("", response_model=list[KnowledgeBase])
async def list_knowledge_bases(claims: Claims, db: DatabaseSession):
    result = await db.scalars(
        select(KnowledgeBase)
        .where(KnowledgeBase.id.in_(accessible_knowledge_bases(_owner_id(claims))))
        .order_by(KnowledgeBase.created_at)
    )
    return list(result)


@router.post("", response_model=KnowledgeBase, status_code=201)
async def create_knowledge_base(
    data: KnowledgeBaseInput, claims: Claims, db: DatabaseSession
):
    item = KnowledgeBase(owner_id=_owner_id(claims), **data.model_dump())
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.get("/{identifier}", response_model=KnowledgeBase)
async def get_knowledge_base(identifier: UUID, claims: Claims, db: DatabaseSession):
    return await require(db, claims, identifier)


@router.patch("/{identifier}", response_model=KnowledgeBase)
async def update_knowledge_base(
    identifier: UUID, data: KnowledgeBaseInput, claims: Claims, db: DatabaseSession
):
    item = await require(db, claims, identifier, "manage")
    item.name, item.description, item.updated_at = (
        data.name,
        data.description,
        utc_now(),
    )
    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{identifier}", status_code=204)
async def delete_knowledge_base(identifier: UUID, claims: Claims, db: DatabaseSession):
    item = await require(db, claims, identifier, "own")
    if item.is_default:
        raise HTTPException(409, "默认知识库不能删除")
    item.is_deleted, item.updated_at = True, utc_now()
    await db.commit()


@router.get("/{identifier}/members", response_model=list[KnowledgeBaseMember])
async def list_members(identifier: UUID, claims: Claims, db: DatabaseSession):
    await require(db, claims, identifier, "manage")
    return list(
        await db.scalars(
            select(KnowledgeBaseMember)
            .where(KnowledgeBaseMember.knowledge_base_id == identifier)
            .order_by(KnowledgeBaseMember.user_id)
        )
    )


@router.put("/{identifier}/members/{user_id}", status_code=204)
async def set_member(
    identifier: UUID,
    user_id: str,
    data: MemberInput,
    claims: Claims,
    db: DatabaseSession,
):
    item = await require(db, claims, identifier, "manage")
    if not user_id.strip() or len(user_id) > 128 or user_id == item.owner_id:
        raise HTTPException(422, "成员身份无效；所有者不通过成员角色修改")
    statement = insert(KnowledgeBaseMember).values(
        knowledge_base_id=identifier,
        user_id=user_id,
        role=data.role,
        updated_at=utc_now(),
    )
    await db.execute(
        statement.on_conflict_do_update(
            index_elements=["knowledge_base_id", "user_id"],
            set_={
                "role": statement.excluded.role,
                "updated_at": statement.excluded.updated_at,
            },
        )
    )
    await db.commit()


@router.delete("/{identifier}/members/{user_id}", status_code=204)
async def remove_member(
    identifier: UUID, user_id: str, claims: Claims, db: DatabaseSession
):
    item = await require(db, claims, identifier, "manage")
    if user_id == item.owner_id:
        raise HTTPException(422, "不能删除所有者")
    await db.execute(
        delete(KnowledgeBaseMember).where(
            KnowledgeBaseMember.knowledge_base_id == identifier,
            KnowledgeBaseMember.user_id == user_id,
        )
    )
    await db.commit()
