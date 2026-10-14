from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from zhiyin.db.knowledge_bases import document_access
from zhiyin.db.models import (
    Chunk,
    ConversationItem,
    Document,
    ItemRole,
    ItemType,
    Session,
    Turn,
    TurnStatus,
    utc_now,
)


@dataclass
class ConversationRepository:
    """持久化会话聚合；事务边界由调用方负责。"""

    _db: AsyncSession

    async def get_session(
        self,
        session_id: UUID,
        owner_id: str,
        *,
        for_update: bool = False,
    ) -> Session | None:
        statement = select(Session).where(
            col(Session.id) == session_id,
            col(Session.owner_id) == owner_id,
            col(Session.is_deleted).is_(False),
        )
        if for_update:
            statement = statement.with_for_update()
        return await self._db.scalar(statement)

    async def create_session(self, owner_id: str) -> Session:
        session = Session(owner_id=owner_id)
        self._db.add(session)
        await self._db.flush()
        return session

    async def create_turn(self, session_id: UUID) -> Turn:
        latest_sequence = await self._db.scalar(
            select(func.max(Turn.sequence)).where(
                col(Turn.session_id) == session_id,
            )
        )
        turn = Turn(
            session_id=session_id,
            sequence=0 if latest_sequence is None else latest_sequence + 1,
        )
        self._db.add(turn)
        await self._db.flush()
        return turn

    async def append_items(
        self,
        items: list[ConversationItem],
    ) -> list[ConversationItem]:
        self._db.add_all(items)
        await self._db.flush()
        return items

    async def list_message_history(
        self,
        session_id: UUID,
        *,
        before_turn_sequence: int,
        after_turn_id: UUID | None = None,
    ) -> list[ConversationItem]:
        statement = (
            select(ConversationItem)
            .join(Turn, col(ConversationItem.turn_id) == col(Turn.id))
            .where(
                col(Turn.session_id) == session_id,
                col(Turn.sequence) < before_turn_sequence,
                or_(
                    col(Turn.status) == TurnStatus.COMPLETED,
                    and_(
                        col(Turn.status) == TurnStatus.FAILED,
                        col(ConversationItem.role) == ItemRole.USER,
                    ),
                ),
                col(ConversationItem.item_type) == ItemType.MESSAGE,
            )
            .order_by(
                col(Turn.sequence),
                col(ConversationItem.sequence),
            )
        )
        if after_turn_id is not None:
            covered_sequence = (
                select(Turn.sequence)
                .where(
                    col(Turn.id) == after_turn_id, col(Turn.session_id) == session_id
                )
                .correlate(None)
                .scalar_subquery()
            )
            statement = statement.where(col(Turn.sequence) > covered_sequence)
        result = await self._db.scalars(statement)
        return list(result)

    async def list_context_history(
        self,
        session_id: UUID,
        *,
        before_turn_sequence: int,
    ) -> list[ConversationItem]:
        """返回最新持久化摘要和覆盖边界之后的问答，原始历史不删除。"""
        statement = (
            select(ConversationItem)
            .join(Turn, col(ConversationItem.turn_id) == col(Turn.id))
            .where(
                col(Turn.session_id) == session_id,
                col(Turn.sequence) < before_turn_sequence,
                col(ConversationItem.item_type) == ItemType.COMPACTION,
            )
            .order_by(col(Turn.sequence).desc(), col(ConversationItem.sequence).desc())
            .limit(1)
        )
        checkpoint = await self._db.scalar(statement)
        after_turn_id = None
        if checkpoint is not None:
            after_turn_id = UUID(str(checkpoint.payload["covered_through_turn_id"]))
        history = await self.list_message_history(
            session_id,
            before_turn_sequence=before_turn_sequence,
            after_turn_id=after_turn_id,
        )
        return [checkpoint, *history] if checkpoint is not None else history

    async def complete_turn(
        self,
        turn: Turn,
        *,
        provider: str | None,
        model: str | None,
        response_id: str | None,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        completed_at: datetime,
    ) -> None:
        turn.status = TurnStatus.COMPLETED
        turn.provider = provider
        turn.model = model
        turn.final_provider_response_id = response_id
        turn.input_tokens = input_tokens
        turn.output_tokens = output_tokens
        turn.total_tokens = total_tokens
        turn.completed_at = completed_at
        await self._db.flush()

    async def fail_turn(
        self,
        turn: Turn,
        *,
        error_message: str,
        completed_at: datetime,
    ) -> None:
        turn.status = TurnStatus.FAILED
        turn.error_message = error_message
        turn.completed_at = completed_at
        await self._db.flush()

    async def list_sessions(
        self,
        owner_id: str,
        *,
        limit: int,
        before: tuple[datetime, UUID] | None = None,
    ) -> list[Session]:
        statement = select(Session).where(
            col(Session.owner_id) == owner_id,
            col(Session.is_deleted).is_(False),
        )
        if before is not None:
            updated_at, session_id = before
            statement = statement.where(
                or_(
                    col(Session.updated_at) < updated_at,
                    and_(
                        col(Session.updated_at) == updated_at,
                        col(Session.id) < session_id,
                    ),
                )
            )
        result = await self._db.scalars(
            statement.order_by(
                col(Session.updated_at).desc(), col(Session.id).desc()
            ).limit(limit)
        )
        return list(result)

    async def list_turns(
        self,
        session_id: UUID,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[Turn]:
        statement = select(Turn).where(col(Turn.session_id) == session_id)
        if before_sequence is not None:
            statement = statement.where(col(Turn.sequence) < before_sequence)
        result = await self._db.scalars(
            statement.order_by(col(Turn.sequence).desc()).limit(limit)
        )
        return list(result)

    async def list_display_items(self, turn_ids: list[UUID]) -> list[ConversationItem]:
        if not turn_ids:
            return []
        result = await self._db.scalars(
            select(ConversationItem)
            .where(
                col(ConversationItem.turn_id).in_(turn_ids),
                col(ConversationItem.item_type).in_(
                    [
                        ItemType.MESSAGE,
                        ItemType.FUNCTION_CALL,
                        ItemType.FUNCTION_CALL_OUTPUT,
                    ]
                ),
            )
            .order_by(col(ConversationItem.sequence))
        )
        return list(result)

    async def list_source_chunks(
        self, chunk_ids: list[UUID], owner_id: str
    ) -> list[tuple[Chunk, str]]:
        if not chunk_ids:
            return []
        result = await self._db.execute(
            select(Chunk, Document.name)
            .join(Document, col(Chunk.document_id) == col(Document.id))
            .where(col(Chunk.id).in_(chunk_ids), document_access(owner_id))
        )
        return [(chunk, name) for chunk, name in result]

    async def touch_session(self, session_id: UUID) -> None:
        await self._db.execute(
            update(Session)
            .where(col(Session.id) == session_id)
            .values(updated_at=utc_now())
        )

    async def delete_session(self, session: Session) -> None:
        session.is_deleted = True
        session.updated_at = utc_now()
        await self._db.flush()
