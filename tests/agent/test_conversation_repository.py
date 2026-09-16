from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.agent.repository import ConversationRepository
from interview_ai.db.models import ConversationItem, ItemRole, ItemType

SESSION_ID = UUID("01900000-0000-7000-8000-000000000010")
TURN_ID = UUID("01900000-0000-7000-8000-000000000011")


@pytest.mark.asyncio
async def test_create_session_flushes_without_committing() -> None:
    db = MagicMock(spec=AsyncSession)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    repository = ConversationRepository(db)

    session = await repository.create_session("user-001")

    assert session.owner_id == "user-001"
    db.add.assert_called_once_with(session)
    db.flush.assert_awaited_once_with()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_message_history_returns_query_result_in_statement_order() -> None:
    db = MagicMock(spec=AsyncSession)
    first = ConversationItem(
        turn_id=TURN_ID,
        sequence=0,
        item_type=ItemType.MESSAGE,
        role=ItemRole.USER,
        text_content="第一问",
    )
    second = ConversationItem(
        turn_id=TURN_ID,
        sequence=1,
        item_type=ItemType.MESSAGE,
        role=ItemRole.ASSISTANT,
        text_content="第一答",
    )
    db.scalars = AsyncMock(return_value=[first, second])
    repository = ConversationRepository(db)

    result = await repository.list_message_history(
        SESSION_ID,
        before_turn_sequence=2,
    )

    assert result == [first, second]
    statement = db.scalars.await_args.args[0]
    sql = str(statement)
    assert "turns.sequence <" in sql
    assert "turns.status =" in sql
    assert "conversation_items.item_type =" in sql
    assert len(statement._order_by_clauses) == 2
