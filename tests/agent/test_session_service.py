from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from interview_ai.agent.repository import ConversationRepository
from interview_ai.agent.services import SessionNotFoundError
from interview_ai.agent.services.session import InvalidCursorError, SessionService
from interview_ai.db.models import (
    Chunk,
    ConversationItem,
    ItemRole,
    ItemType,
    Session,
    Turn,
    TurnStatus,
)


def make_service():
    db = MagicMock(spec=AsyncSession)
    repository = AsyncMock(spec=ConversationRepository)
    return SessionService(db, repository), db, repository


@pytest.mark.asyncio
async def test_sessions_cursor_round_trip_and_last_page():
    service, _, repository = make_service()
    timestamp = datetime(2026, 10, 10, tzinfo=UTC)
    sessions = [Session(owner_id="owner", updated_at=timestamp) for _ in range(3)]
    repository.list_sessions.return_value = sessions
    page = await service.list_sessions("owner", limit=2)
    assert page.items == sessions[:2]
    assert page.next_cursor is not None
    repository.list_sessions.assert_awaited_once_with("owner", limit=3, before=None)
    repository.list_sessions.return_value = sessions[2:]
    last = await service.list_sessions("owner", limit=2, cursor=page.next_cursor)
    repository.list_sessions.assert_awaited_with(
        "owner", limit=3, before=(timestamp, sessions[1].id)
    )
    assert last.items == sessions[2:]
    assert last.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "!!!!",
        "bm90LWEtdGltZXxub3QtYS11dWlk",
        "MjAyNi0xMC0xMHwwMTkwMDAwMC0wMDAwLTcwMDAtODAwMC0wMDAwMDAwMDAwMTA=",
    ],
)
async def test_invalid_cursor_does_not_query_database(cursor):
    service, _, repository = make_service()
    with pytest.raises(InvalidCursorError):
        await service.list_sessions("owner", cursor=cursor)
    repository.list_sessions.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_session_page():
    service, _, repository = make_service()
    repository.list_sessions.return_value = []
    page = await service.list_sessions("owner")
    assert page.items == []
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_turn_page_is_chronological_and_keeps_each_turn_complete():
    service, _, repository = make_service()
    session = Session(owner_id="owner")
    repository.get_session.return_value = session
    turns = [
        Turn(session_id=session.id, sequence=i, status=status)
        for i, status in [
            (2, TurnStatus.IN_PROGRESS),
            (1, TurnStatus.FAILED),
            (0, TurnStatus.COMPLETED),
        ]
    ]
    repository.list_turns.return_value = turns
    user = ConversationItem(
        turn_id=turns[1].id,
        sequence=0,
        item_type=ItemType.MESSAGE,
        role=ItemRole.USER,
        text_content="问题",
    )
    assistant = ConversationItem(
        turn_id=turns[1].id,
        sequence=2,
        item_type=ItemType.MESSAGE,
        role=ItemRole.ASSISTANT,
        text_content="答案",
    )
    repository.list_display_items.return_value = [assistant, user]
    page = await service.list_turns(session.id, "owner", limit=2, cursor=3)
    assert [record.turn.sequence for record in page.items] == [1, 2]
    assert page.items[0].items == [user, assistant]
    assert page.items[1].items == []
    assert page.next_cursor == 1
    repository.list_turns.assert_awaited_once_with(
        session.id, limit=3, before_sequence=3
    )
    repository.list_display_items.assert_awaited_once_with([turns[0].id, turns[1].id])
    repository.list_source_chunks.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_turn_page():
    service, _, repository = make_service()
    repository.get_session.return_value = Session(owner_id="owner")
    repository.list_turns.return_value = []
    repository.list_display_items.return_value = []
    page = await service.list_turns(uuid7(), "owner")
    assert page.items == []
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_legacy_sources_are_restored_in_one_query_without_mutating_records():
    service, _, repository = make_service()
    session = Session(owner_id="owner")
    turn = Turn(session_id=session.id, sequence=0)
    chunk = Chunk(
        id=uuid7(),
        document_id=uuid7(),
        sort_index=0,
        level=1,
        headings=("标题",),
        content="资料",
        start_line=1,
        end_line=2,
        hash="hash",
    )
    missing_id = uuid7()
    item = ConversationItem(
        turn_id=turn.id,
        sequence=1,
        item_type=ItemType.FUNCTION_CALL_OUTPUT,
        payload={
            "sources": [
                {"chunk_id": str(chunk.id), "score": 0.9},
                {"chunk_id": str(missing_id), "score": 0.5},
            ]
        },
    )
    repository.get_session.return_value = session
    repository.list_turns.return_value = [turn]
    repository.list_display_items.return_value = [item]
    repository.list_source_chunks.return_value = [(chunk, "文档")]
    page = await service.list_turns(session.id, "owner")
    sources = page.items[0].items[0].payload["sources"]
    assert len(sources) == 1
    assert sources[0]["content"] == "资料"
    assert sources[0]["document_name"] == "文档"
    assert sources[0]["score"] == 0.9
    assert sources[0]["headings"] == ["标题"]
    assert "content" not in item.payload["sources"][0]
    assert repository.list_source_chunks.await_count == 1
    ids, owner = repository.list_source_chunks.await_args.args
    assert set(ids) == {chunk.id, missing_id}
    assert owner == "owner"


@pytest.mark.asyncio
async def test_snapshot_sources_do_not_require_existing_document():
    service, _, repository = make_service()
    snapshot = {"chunk_id": str(uuid7()), "content": "原始来源", "score": 0.9}
    item = ConversationItem(
        turn_id=uuid7(),
        sequence=1,
        item_type=ItemType.FUNCTION_CALL_OUTPUT,
        payload={"sources": [snapshot]},
    )
    result = await service._restore_sources([item], "owner")
    assert result[0].payload["sources"] == [snapshot]
    repository.list_source_chunks.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "turns", "delete"])
async def test_missing_or_inaccessible_session_prevents_content_access(operation):
    service, _, repository = make_service()
    repository.get_session.return_value = None
    session_id = uuid7()
    with pytest.raises(SessionNotFoundError):
        if operation == "get":
            await service.get_session(session_id, "other-owner")
        elif operation == "turns":
            await service.list_turns(session_id, "other-owner")
        else:
            await service.delete_session(session_id, "other-owner")
    repository.list_turns.assert_not_awaited()
    repository.list_display_items.assert_not_awaited()
    repository.delete_session.assert_not_awaited()
    args, kwargs = repository.get_session.await_args
    assert args == (session_id, "other-owner")
    assert kwargs == ({"for_update": True} if operation == "delete" else {})


@pytest.mark.asyncio
async def test_delete_uses_short_transaction_and_owned_session_lock():
    service, db, repository = make_service()
    session = Session(owner_id="owner")
    repository.get_session.return_value = session
    await service.delete_session(session.id, "owner")
    assert db.begin.call_count == 1
    repository.get_session.assert_awaited_once_with(
        session.id, "owner", for_update=True
    )
    repository.delete_session.assert_awaited_once_with(session)
    db.begin.return_value.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_loading_older_turn_pages_has_no_duplicates_or_missing_turns():
    service, _, repository = make_service()
    session = Session(owner_id="owner")
    turns = [Turn(session_id=session.id, sequence=i) for i in range(5)]
    repository.get_session.return_value = session
    repository.list_display_items.return_value = []

    async def list_turns(session_id, *, limit, before_sequence):
        eligible = [
            turn
            for turn in reversed(turns)
            if before_sequence is None or turn.sequence < before_sequence
        ]
        return eligible[:limit]

    repository.list_turns.side_effect = list_turns
    cursor = None
    sequences = []
    while True:
        page = await service.list_turns(session.id, "owner", limit=2, cursor=cursor)
        sequences = [record.turn.sequence for record in page.items] + sequences
        cursor = page.next_cursor
        if cursor is None:
            break
    assert sequences == [0, 1, 2, 3, 4]
    assert repository.list_turns.await_count == 3
