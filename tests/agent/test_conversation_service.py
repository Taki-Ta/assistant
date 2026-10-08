import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from anyio import CancelScope
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from interview_ai.agent.context.builder import ContextBuilder
from interview_ai.agent.context.models import BuiltContext, CompactionCheckpoint
from interview_ai.agent.models import (
    AgentEvent,
    AgentMessage,
    AssistantMessageEvent,
    AssistantTextDeltaEvent,
    ChatCompletedEvent,
    ChatStartedEvent,
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
    repository.list_context_history.return_value = []
    provider = MagicMock()
    provider.tool_definitions.return_value = []
    service = ConversationService(db, repository, provider, ContextBuilder())
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
    repository.list_context_history.assert_awaited_once_with(
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
    repository.list_context_history.return_value = [
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

    repository.list_context_history.assert_awaited_once_with(
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
    assert provider.generate.call_args.kwargs == {
        "instructions": service._context_builder.instructions,
        "stream": False,
    }
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["build", "generate", "persist"])
async def test_chat_cancellation_marks_turn_failed_and_propagates(stage) -> None:
    service, db, repository, provider = _service()
    repository.create_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    turn = Turn(id=TURN_ID, session_id=SESSION_ID, sequence=0)
    repository.create_turn.return_value = turn
    started = asyncio.Event()

    async def wait_for_cancellation():
        started.set()
        await asyncio.Event().wait()

    if stage == "build":
        service._context_builder = MagicMock()

        async def build(*args):
            await wait_for_cancellation()

        service._context_builder.build = AsyncMock(side_effect=build)

    async def generate(*args, **kwargs):
        yield FunctionCallEvent(
            call_id="call-1",
            tool_name="search_knowledge",
            arguments={"query": "Python"},
        )
        if stage == "generate":
            await wait_for_cancellation()
        yield AssistantMessageEvent(text="回答")
        yield _completed()

    provider.generate.side_effect = generate
    if stage == "persist":
        append_count = 0

        async def append_items(items):
            nonlocal append_count
            append_count += 1
            if append_count == 2:
                await wait_for_cancellation()

        repository.append_items.side_effect = append_items

    context, dependencies = _runtime()
    task = asyncio.create_task(service.chat(None, "用户问题", context, dependencies))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    repository.fail_turn.assert_awaited_once()
    failure = repository.fail_turn.await_args
    assert failure.args == (turn,)
    assert failure.kwargs["error_message"] == "请求已取消"
    assert failure.kwargs["completed_at"] is not None
    repository.complete_turn.assert_not_awaited()
    assert db.begin.call_count == (3 if stage == "persist" else 2)
    user_item = repository.append_items.await_args_list[0].args[0][0]
    assert user_item.text_content == "用户问题"
    if stage != "build":
        saved_events = repository.append_items.await_args_list[-1].args[0]
        assert saved_events[0].item_type == ItemType.FUNCTION_CALL
        assert saved_events[0].sequence == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_compaction_is_saved_before_generation_even_when_generation_fails(fails):
    service, db, repository, provider = _service()
    repository.get_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    repository.create_turn.return_value = Turn(
        id=TURN_ID, session_id=SESSION_ID, sequence=2
    )
    checkpoint = CompactionCheckpoint(text="历史摘要", covered_through_turn_id=uuid7())
    service._context_builder = MagicMock()
    service._context_builder.build = AsyncMock(
        return_value=BuiltContext(
            instructions="规则",
            messages=[AgentMessage(role="user", content="继续")],
            estimated_input_tokens=100,
            compaction=checkpoint,
        )
    )

    def generate(*args, **kwargs):
        saved = repository.append_items.await_args_list[1].args[0][0]
        assert saved.item_type == ItemType.COMPACTION
        assert saved.sequence == 1
        assert saved.payload["covered_through_turn_id"] == str(
            checkpoint.covered_through_turn_id
        )
        return _events(
            AssistantMessageEvent(text="回答"),
            *([] if fails else [_completed()]),
            error=RuntimeError("模型失败") if fails else None,
        )

    provider.generate.side_effect = generate
    context, dependencies = _runtime()
    if fails:
        with pytest.raises(RuntimeError, match="模型失败"):
            await service.chat(SESSION_ID, "继续", context, dependencies)
        repository.fail_turn.assert_awaited_once()
    else:
        await service.chat(SESSION_ID, "继续", context, dependencies)
        repository.complete_turn.assert_awaited_once()
    assert db.begin.call_count == 3
    assert repository.append_items.await_args_list[-1].args[0][0].sequence == 2


@pytest.mark.asyncio
async def test_chat_stream_sends_deltas_before_persisting_and_completes_after_commit():
    service, db, repository, provider = _service()
    repository.create_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    repository.create_turn.return_value = Turn(
        id=TURN_ID, session_id=SESSION_ID, sequence=0
    )
    provider.generate.return_value = _events(
        FunctionCallEvent(
            call_id="call-1", tool_name="search", arguments={"query": "问题"}
        ),
        FunctionCallOutputEvent(
            call_id="call-1",
            tool_name="search",
            output="[]",
            succeeded=True,
            sources=(_source(CHUNK_A, 0.9),),
        ),
        AssistantTextDeltaEvent(delta="回答"),
        AssistantMessageEvent(text="回答"),
        _completed(),
    )
    context, dependencies = _runtime()
    events = service.chat_stream(None, "问题", context, dependencies)
    first = await anext(events)
    assert isinstance(first, ChatStartedEvent)
    assert first.session_id == SESSION_ID and first.turn_id == TURN_ID
    provider.generate.assert_not_called()
    assert repository.append_items.await_count == 1
    assert db.begin.return_value.__aexit__.await_count == 1
    call = await anext(events)
    assert isinstance(call, FunctionCallEvent)
    assert call.arguments == {"query": "问题"}
    output = await anext(events)
    assert isinstance(output, FunctionCallOutputEvent)
    assert output.call_id == call.call_id
    assert output.sources == (_source(CHUNK_A, 0.9),)
    repository.complete_turn.assert_not_awaited()
    delta = await anext(events)
    assert isinstance(delta, AssistantTextDeltaEvent)
    repository.complete_turn.assert_not_awaited()
    final = await anext(events)
    assert isinstance(final, ChatCompletedEvent)
    assert final.answer == "回答"
    assert final.retrieved_sources == (_source(CHUNK_A, 0.9),)
    repository.complete_turn.assert_awaited_once()
    assert db.begin.return_value.__aexit__.await_count == 2
    saved = repository.append_items.await_args_list[-1].args[0]
    assert [item.item_type for item in saved] == [
        ItemType.FUNCTION_CALL,
        ItemType.FUNCTION_CALL_OUTPUT,
        ItemType.MESSAGE,
    ]
    assert provider.generate.call_args.kwargs["stream"] is True
    await events.aclose()
    repository.fail_turn.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_delta", [False, True])
async def test_closing_chat_stream_marks_failed_and_closes_provider(after_delta):
    service, _, repository, provider = _service()
    repository.create_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    repository.create_turn.return_value = Turn(
        id=TURN_ID, session_id=SESSION_ID, sequence=0
    )
    closed = False

    async def generate(*args, **kwargs):
        nonlocal closed
        try:
            yield AssistantTextDeltaEvent(delta="部分")
            await asyncio.Event().wait()
        finally:
            closed = True

    provider.generate.side_effect = generate
    context, dependencies = _runtime()
    events = service.chat_stream(None, "原始问题", context, dependencies)
    await anext(events)
    if after_delta:
        await anext(events)
    await events.aclose()
    assert closed == after_delta
    repository.fail_turn.assert_awaited_once()
    assert repository.fail_turn.await_args.kwargs["error_message"] == "请求已取消"
    assert repository.append_items.await_count == 1
    repository.complete_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_failure_does_not_save_text_deltas_or_emit_completed():
    service, _, repository, provider = _service()
    repository.create_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    repository.create_turn.return_value = Turn(
        id=TURN_ID, session_id=SESSION_ID, sequence=0
    )
    provider.generate.return_value = _events(
        AssistantTextDeltaEvent(delta="部分"), error=RuntimeError("失败")
    )
    context, dependencies = _runtime()
    received = []
    with pytest.raises(RuntimeError, match="失败"):
        async for event in service.chat_stream(None, "问题", context, dependencies):
            received.append(event)
    assert [type(event) for event in received] == [
        ChatStartedEvent,
        AssistantTextDeltaEvent,
    ]
    repository.fail_turn.assert_awaited_once()
    repository.complete_turn.assert_not_awaited()
    assert repository.append_items.await_count == 1


@pytest.mark.asyncio
async def test_stream_cleanup_is_shielded_from_cancel_scope():
    service, _, repository, _ = _service()
    repository.create_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    repository.create_turn.return_value = Turn(
        id=TURN_ID, session_id=SESSION_ID, sequence=0
    )
    cleaned = False

    async def fail_turn(*args, **kwargs):
        nonlocal cleaned
        await asyncio.sleep(0)
        cleaned = True

    repository.fail_turn.side_effect = fail_turn
    context, dependencies = _runtime()
    events = service.chat_stream(None, "问题", context, dependencies)
    await anext(events)
    with CancelScope() as scope:
        scope.cancel()
        await events.aclose()
    assert cleaned


@pytest.mark.asyncio
async def test_stream_does_not_emit_completed_when_persistence_fails():
    service, _, repository, provider = _service()
    repository.create_session.return_value = Session(id=SESSION_ID, owner_id="user-001")
    repository.create_turn.return_value = Turn(
        id=TURN_ID, session_id=SESSION_ID, sequence=0
    )
    provider.generate.return_value = _events(
        AssistantMessageEvent(text="回答"), _completed()
    )
    repository.complete_turn.side_effect = RuntimeError("入库失败")
    context, dependencies = _runtime()
    received = []
    with pytest.raises(RuntimeError, match="入库失败"):
        async for event in service.chat_stream(None, "问题", context, dependencies):
            received.append(event)
    assert [type(event) for event in received] == [ChatStartedEvent]
    repository.fail_turn.assert_awaited_once()
