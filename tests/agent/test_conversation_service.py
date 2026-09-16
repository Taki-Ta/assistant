from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.agent.models import (
    AgentEvent,
    AgentMessage,
    AssistantMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
    RetrievedSource,
)
from interview_ai.agent.repository import ConversationRepository
from interview_ai.agent.runtime import AgentContext, ToolDependencies
from interview_ai.agent.services import ConversationService, SessionNotFoundError
from interview_ai.db.models import ConversationItem, ItemRole, ItemType, Session, Turn
from interview_ai.indexing.search_service import SearchService

SESSION_ID = UUID("01900000-0000-7000-8000-000000000010")
TURN_ID = UUID("01900000-0000-7000-8000-000000000011")
CHUNK_A = UUID("01900000-0000-7000-8000-000000000021")
CHUNK_B = UUID("01900000-0000-7000-8000-000000000022")


def _service() -> tuple[
    ConversationService,
    MagicMock,
    AsyncMock,
    MagicMock,
]:
    db = MagicMock(spec=AsyncSession)
    repository = AsyncMock(spec=ConversationRepository)
    repository.list_message_history.return_value = []
    provider = MagicMock()
    service = ConversationService(db, repository, provider)
    return service, db, repository, provider


def _runtime() -> tuple[AgentContext, ToolDependencies]:
    return (
        AgentContext(owner_id="user-001"),
        ToolDependencies(search_service=AsyncMock(spec=SearchService)),
    )


def _events(
    *events: AgentEvent,
    error: Exception | None = None,
) -> AsyncIterator[AgentEvent]:
    async def stream() -> AsyncIterator[AgentEvent]:
        for event in events:
            yield event
        if error is not None:
            raise error

    return stream()


def _completed() -> ModelCompletedEvent:
    return ModelCompletedEvent(
        provider_response_id="resp-001",
        provider="openai",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )


def _source(chunk_id: UUID, score: float) -> RetrievedSource:
    return RetrievedSource(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        score=score,
    )


@pytest.mark.asyncio
async def test_chat_uses_two_short_transactions() -> None:
    service, db, repository, provider = _service()
    session = Session(id=SESSION_ID, owner_id="user-001")
    turn = Turn(id=TURN_ID, session_id=SESSION_ID, sequence=0)
    repository.create_session.return_value = session
    repository.create_turn.return_value = turn
    provider.generate.return_value = _events(
        AssistantMessageEvent(text="最终回答"),
        _completed(),
    )
    context, dependencies = _runtime()

    result = await service.chat(None, "用户问题", context, dependencies)

    assert result.session_id == SESSION_ID
    assert result.answer == "最终回答"
    assert result.retrieved_sources == ()
    assert db.begin.call_count == 2
    repository.create_session.assert_awaited_once_with("user-001")
    repository.create_turn.assert_awaited_once_with(SESSION_ID)
    repository.list_message_history.assert_awaited_once_with(
        SESSION_ID,
        before_turn_sequence=0,
    )
    assert repository.append_items.await_count == 2
    user_item = repository.append_items.await_args_list[0].args[0][0]
    assistant_item = repository.append_items.await_args_list[1].args[0][0]
    assert user_item.role == ItemRole.USER
    assert assistant_item.role == ItemRole.ASSISTANT
    repository.complete_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_passes_completed_message_history_to_provider() -> None:
    service, _, repository, provider = _service()
    session = Session(id=SESSION_ID, owner_id="user-001")
    turn = Turn(id=TURN_ID, session_id=SESSION_ID, sequence=2)
    repository.get_session.return_value = session
    repository.create_turn.return_value = turn
    repository.list_message_history.return_value = [
        ConversationItem(
            turn_id=TURN_ID,
            sequence=0,
            item_type=ItemType.MESSAGE,
            role=ItemRole.USER,
            text_content="第一问",
        ),
        ConversationItem(
            turn_id=TURN_ID,
            sequence=1,
            item_type=ItemType.MESSAGE,
            role=ItemRole.ASSISTANT,
            text_content="第一答",
        ),
    ]
    provider.generate.return_value = _events(
        AssistantMessageEvent(text="第二答"),
        _completed(),
    )
    context, dependencies = _runtime()

    await service.chat(SESSION_ID, "第二问", context, dependencies)

    repository.list_message_history.assert_awaited_once_with(
        SESSION_ID,
        before_turn_sequence=2,
    )
    assert provider.generate.call_args.args == (
        [
            AgentMessage(role="user", content="第一问"),
            AgentMessage(role="assistant", content="第一答"),
            AgentMessage(role="user", content="第二问"),
        ],
        context,
        dependencies,
    )
    current_user_item = repository.append_items.await_args_list[0].args[0][0]
    assert current_user_item.text_content == "第二问"
    assert current_user_item.payload == {}


@pytest.mark.asyncio
async def test_chat_unions_all_successful_retrievals_and_keeps_highest_score() -> None:
    service, _, repository, provider = _service()
    session = Session(id=SESSION_ID, owner_id="user-001")
    turn = Turn(id=TURN_ID, session_id=SESSION_ID, sequence=0)
    repository.create_session.return_value = session
    repository.create_turn.return_value = turn
    source_a_low = _source(CHUNK_A, 0.7)
    source_a_high = _source(CHUNK_A, 0.9)
    source_b = _source(CHUNK_B, 0.8)
    provider.generate.return_value = _events(
        FunctionCallEvent(
            call_id="call-1",
            tool_name="search_knowledge",
            arguments={"query": "first"},
        ),
        FunctionCallOutputEvent(
            call_id="call-1",
            tool_name="search_knowledge",
            output="first output",
            succeeded=True,
            sources=(source_a_low, source_b),
        ),
        FunctionCallEvent(
            call_id="call-2",
            tool_name="search_knowledge",
            arguments={"query": "second"},
        ),
        FunctionCallOutputEvent(
            call_id="call-2",
            tool_name="search_knowledge",
            output="second output",
            succeeded=True,
            sources=(source_a_high,),
        ),
        AssistantMessageEvent(text="综合回答"),
        _completed(),
    )
    context, dependencies = _runtime()

    result = await service.chat(None, "用户问题", context, dependencies)

    assert result.retrieved_sources == (source_a_high, source_b)
    saved_items = repository.append_items.await_args_list[1].args[0]
    assert [item.sequence for item in saved_items] == [1, 2, 3, 4, 5]
    assert [item.item_type for item in saved_items] == [
        ItemType.FUNCTION_CALL,
        ItemType.FUNCTION_CALL_OUTPUT,
        ItemType.FUNCTION_CALL,
        ItemType.FUNCTION_CALL_OUTPUT,
        ItemType.MESSAGE,
    ]
    assert saved_items[1].payload["sources"][0]["chunk_id"] == str(CHUNK_A)


@pytest.mark.asyncio
async def test_chat_rejects_missing_or_foreign_session() -> None:
    service, db, repository, provider = _service()
    repository.get_session.return_value = None
    context, dependencies = _runtime()

    with pytest.raises(SessionNotFoundError, match=str(SESSION_ID)):
        await service.chat(SESSION_ID, "用户问题", context, dependencies)

    assert db.begin.call_count == 1
    repository.get_session.assert_awaited_once_with(
        SESSION_ID,
        "user-001",
        for_update=True,
    )
    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_chat_persists_partial_events_and_marks_turn_failed() -> None:
    service, db, repository, provider = _service()
    session = Session(id=SESSION_ID, owner_id="user-001")
    turn = Turn(id=TURN_ID, session_id=SESSION_ID, sequence=0)
    repository.create_session.return_value = session
    repository.create_turn.return_value = turn
    provider.generate.return_value = _events(
        FunctionCallEvent(
            call_id="call-1",
            tool_name="search_knowledge",
            arguments={"query": "Python"},
        ),
        error=RuntimeError("model unavailable"),
    )
    context, dependencies = _runtime()

    with pytest.raises(RuntimeError, match="model unavailable"):
        await service.chat(None, "用户问题", context, dependencies)

    assert db.begin.call_count == 2
    assert repository.append_items.await_count == 2
    partial_item = repository.append_items.await_args_list[1].args[0][0]
    assert partial_item.item_type == ItemType.FUNCTION_CALL
    repository.fail_turn.assert_awaited_once()
    repository.complete_turn.assert_not_awaited()
