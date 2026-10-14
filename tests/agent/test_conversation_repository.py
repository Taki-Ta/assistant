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
    assert "conversation_items.role =" in sql
    assert statement.compile().params["status_1"] == "completed"
    assert statement.compile().params["status_2"] == "failed"
    assert statement.compile().params["role_1"] == "user"
    assert "conversation_items.item_type =" in sql
    assert len(statement._order_by_clauses) == 2


@pytest.mark.asyncio
async def test_context_history_restores_checkpoint_and_filters_covered_turns():
    db = MagicMock(spec=AsyncSession)
    checkpoint = ConversationItem(
        turn_id=TURN_ID,
        sequence=1,
        item_type=ItemType.COMPACTION,
        role=ItemRole.USER,
        text_content="摘要",
        payload={"covered_through_turn_id": str(TURN_ID)},
    )
    db.scalar = AsyncMock(return_value=checkpoint)
    db.scalars = AsyncMock(return_value=[])
    result = await ConversationRepository(db).list_context_history(
        SESSION_ID,
        before_turn_sequence=3,
    )
    assert result == [checkpoint]
    statement = db.scalars.await_args.args[0]
    assert "turns.sequence > (SELECT turns.sequence" in str(statement)
    assert TURN_ID in statement.compile().params.values()


@pytest.mark.asyncio
async def test_context_history_without_checkpoint_returns_original_history():
    db = MagicMock(spec=AsyncSession)
    db.scalar = AsyncMock(return_value=None)
    db.scalars = AsyncMock(return_value=[])
    assert (
        await ConversationRepository(db).list_context_history(
            SESSION_ID,
            before_turn_sequence=0,
        )
        == []
    )


@pytest.mark.asyncio
async def test_session_queries_enforce_owner_deletion_filter_and_stable_pagination():
    from datetime import UTC, datetime

    db = MagicMock(spec=AsyncSession)
    db.scalars = AsyncMock(return_value=[])
    db.scalar = AsyncMock(return_value=None)
    repository = ConversationRepository(db)
    timestamp = datetime(2026, 10, 10, tzinfo=UTC)
    await repository.list_sessions("owner", limit=21, before=(timestamp, SESSION_ID))
    statement = db.scalars.await_args.args[0]
    sql = str(statement)
    params = statement.compile().params
    assert "sessions.owner_id =" in sql
    assert "sessions.is_deleted IS false" in sql
    assert "sessions.updated_at <" in sql
    assert "sessions.updated_at =" in sql
    assert "sessions.id <" in sql
    assert "ORDER BY sessions.updated_at DESC, sessions.id DESC" in sql
    assert "owner" in params.values()
    assert timestamp in params.values()
    assert SESSION_ID in params.values()
    assert statement._limit_clause.value == 21
    await repository.get_session(SESSION_ID, "other-owner")
    statement = db.scalar.await_args.args[0]
    assert "sessions.is_deleted IS false" in str(statement)
    assert "other-owner" in statement.compile().params.values()
    assert SESSION_ID in statement.compile().params.values()


@pytest.mark.asyncio
async def test_display_history_keeps_all_turn_statuses_and_excludes_internal_items():
    db = MagicMock(spec=AsyncSession)
    db.scalars = AsyncMock(return_value=[])
    repository = ConversationRepository(db)
    await repository.list_turns(SESSION_ID, limit=21, before_sequence=4)
    statement = db.scalars.await_args.args[0]
    assert "turns.sequence <" in str(statement)
    assert "turns.status =" not in str(statement)
    assert "ORDER BY turns.sequence DESC" in str(statement)
    assert statement._limit_clause.value == 21
    assert SESSION_ID in statement.compile().params.values()
    await repository.list_display_items([TURN_ID])
    statement = db.scalars.await_args.args[0]
    values = statement.compile().params
    assert values["turn_id_1"] == [TURN_ID]
    assert set(values["item_type_1"]) == {
        "message",
        "function_call",
        "function_call_output",
    }
    assert "ORDER BY conversation_items.sequence" in str(statement)
    db.scalars.reset_mock()
    assert await repository.list_display_items([]) == []
    db.scalars.assert_not_awaited()


@pytest.mark.asyncio
async def test_soft_delete_preserves_history_and_does_not_commit():
    from interview_ai.db.models import Session

    db = MagicMock(spec=AsyncSession)
    db.flush = AsyncMock()
    repository = ConversationRepository(db)
    session = Session(owner_id="owner")
    previous = session.updated_at
    await repository.delete_session(session)
    assert session.is_deleted is True
    assert session.updated_at >= previous
    db.flush.assert_awaited_once()
    db.delete.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_touch_session_updates_only_timestamp_not_deleted_flag():
    db = MagicMock(spec=AsyncSession)
    db.execute = AsyncMock()
    await ConversationRepository(db).touch_session(SESSION_ID)
    statement = db.execute.await_args.args[0]
    assert "UPDATE sessions SET updated_at=" in str(statement)
    assert "is_deleted" not in str(statement)
    assert SESSION_ID in statement.compile().params.values()


@pytest.mark.asyncio
async def test_legacy_source_lookup_is_scoped_to_knowledge_base_access():
    db = MagicMock(spec=AsyncSession)
    db.execute = AsyncMock(return_value=[])
    repository = ConversationRepository(db)
    assert await repository.list_source_chunks([TURN_ID], "owner") == []
    statement = db.execute.await_args.args[0]
    assert "JOIN documents" in str(statement)
    assert "documents.knowledge_base_id IN" in str(statement)
    assert "knowledge_base_members.user_id =" in str(statement)
    assert "owner" in statement.compile().params.values()
    db.execute.reset_mock()
    assert await repository.list_source_chunks([], "owner") == []
    db.execute.assert_not_awaited()
