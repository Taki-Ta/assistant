from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from interview_ai.db.models import (
    ConversationItem,
    ItemType,
    Session,
    Turn,
    TurnStatus,
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
    ) -> list[ConversationItem]:
        statement = (
            select(ConversationItem)
            .join(Turn, col(ConversationItem.turn_id) == col(Turn.id))
            .where(
                col(Turn.session_id) == session_id,
                col(Turn.sequence) < before_turn_sequence,
                col(Turn.status) == TurnStatus.COMPLETED,
                col(ConversationItem.item_type) == ItemType.MESSAGE,
            )
            .order_by(
                col(Turn.sequence),
                col(ConversationItem.sequence),
            )
        )
        result = await self._db.scalars(statement)
        return list(result)

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
