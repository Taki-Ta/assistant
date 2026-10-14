"""Validated user preferences, stored independently of answer transactions."""

import json
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from zhiyin.db.models import Session, Turn, UserPreference, utc_now

PreferenceKey = Literal["response_language", "answer_detail", "code_language"]
ALLOWED_VALUES = {
    "response_language": {"zh-CN", "en", "ja", "ko", "fr", "de", "es"},
    "answer_detail": {"concise", "balanced", "detailed"},
    "code_language": {
        "Python",
        "JavaScript",
        "TypeScript",
        "Java",
        "Go",
        "Rust",
        "C",
        "C++",
        "C#",
        "SQL",
        "Bash",
        "Kotlin",
        "Swift",
        "PHP",
        "Ruby",
    },
}


class RememberPreferenceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: PreferenceKey
    value: str = Field(max_length=32)
    evidence: str = Field(
        min_length=1, max_length=500, description="当前用户消息中表达长期偏好的原文"
    )

    @model_validator(mode="after")
    def validate_value(self):
        if self.value not in ALLOWED_VALUES[self.key]:
            raise ValueError(f"{self.key} 支持的值：{sorted(ALLOWED_VALUES[self.key])}")
        return self


class ForgetPreferenceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: PreferenceKey
    evidence: str = Field(
        min_length=1, max_length=500, description="当前用户消息中要求删除偏好的原文"
    )


class PreferenceService:
    def __init__(self, session_factory: async_sessionmaker):
        self.session_factory = session_factory

    async def list(self, owner_id: str) -> dict[str, str]:
        async with self.session_factory() as db:
            rows = await db.scalars(
                select(UserPreference).where(UserPreference.owner_id == owner_id)
            )
            return {row.key: row.value for row in rows}

    async def change(
        self,
        owner_id: str,
        turn_id: UUID | None,
        user_input: str,
        args: RememberPreferenceArguments | ForgetPreferenceArguments,
    ) -> str:
        if turn_id is None or args.evidence not in user_input:
            raise ValueError("偏好修改必须引用当前用户消息")
        async with self.session_factory() as db, db.begin():
            source = await db.scalar(
                select(Turn.id)
                .join(Session, Turn.session_id == Session.id)
                .where(
                    Turn.id == turn_id,
                    Session.owner_id == owner_id,
                    Session.is_deleted.is_(False),
                )
            )
            if source is None:
                raise ValueError("偏好来源不属于当前用户")
            if isinstance(args, RememberPreferenceArguments):
                statement = insert(UserPreference).values(
                    owner_id=owner_id,
                    key=args.key,
                    value=args.value,
                    source_turn_id=turn_id,
                    updated_at=utc_now(),
                )
                await db.execute(
                    statement.on_conflict_do_update(
                        index_elements=["owner_id", "key"],
                        set_={
                            "value": statement.excluded.value,
                            "source_turn_id": statement.excluded.source_turn_id,
                            "updated_at": statement.excluded.updated_at,
                        },
                    )
                )
                result = {"status": "saved", "key": args.key, "value": args.value}
            else:
                await db.execute(
                    delete(UserPreference).where(
                        UserPreference.owner_id == owner_id,
                        UserPreference.key == args.key,
                    )
                )
                result = {"status": "deleted", "key": args.key}
        return json.dumps(result, ensure_ascii=False)
