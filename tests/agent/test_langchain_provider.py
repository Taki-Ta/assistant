import asyncio
import json
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from langchain.messages import AIMessage, AIMessageChunk
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from interview_ai.agent.context.budget import ContextBudget, ContextBudgetExceeded
from interview_ai.agent.models import (
    AgentMessage,
    AssistantMessageEvent,
    AssistantTextDeltaEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
)
from interview_ai.agent.providers.langchain import LangChainProvider
from interview_ai.agent.providers.openai import (
    ModelResponseError,
    ToolCallLimitExceeded,
)
from interview_ai.agent.runtime import AgentContext, ToolDependencies
from interview_ai.agent.services.conversation import _event_to_item
from interview_ai.db.models import ItemType
from interview_ai.indexing.models import SearchResult


class ScriptedModel(BaseChatModel):
    """Exercise real LangGraph nodes without any provider/network requests."""

    responses: list[AIMessage]
    inputs: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(**kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.inputs.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator:
        result = self._generate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                **message.model_dump(exclude={"type"}), chunk_position="last"
            )
        )


def model_message(text, index, calls=(), **metadata):
    return AIMessage(
        content=text,
        id=f"lc-message-{index}",
        tool_calls=list(calls),
        response_metadata={"id": f"resp-{index}", **metadata},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def call(index):
    return {
        "id": f"call-{index}",
        "name": "search_knowledge",
        "args": {"query": "Python"},
    }


def runtime(search=None):
    return AgentContext(owner_id="actual-user"), ToolDependencies(
        search_service=SimpleNamespace(search=search or AsyncMock(return_value=[]))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("forget", [False, True])
async def test_preference_tools_receive_authenticated_current_turn(forget):
    preference_service = SimpleNamespace(
        change=AsyncMock(return_value='{"status":"saved"}')
    )
    arguments = {"key": "code_language", "evidence": "以后用 Python"}
    if not forget:
        arguments["value"] = "Python"
    model = ScriptedModel(
        responses=[
            model_message(
                "",
                1,
                [
                    {
                        "id": "preference-1",
                        "name": "forget_preference"
                        if forget
                        else "remember_preference",
                        "args": arguments,
                    }
                ],
            ),
            model_message("已处理", 2),
        ]
    )
    provider = LangChainProvider(model=model)
    context = AgentContext("alice", UUID(int=1), "以后用 Python")
    events = [
        event
        async for event in provider.generate(
            [AgentMessage("user", context.user_input)],
            context,
            ToolDependencies(SimpleNamespace(), preference_service),
        )
    ]
    args = preference_service.change.await_args.args
    assert args[:3] == ("alice", UUID(int=1), "以后用 Python")
    assert args[3].key == "code_language"
    assert next(
        event for event in events if isinstance(event, FunctionCallOutputEvent)
    ).succeeded


@pytest.mark.asyncio
async def test_real_agent_tool_loop_maps_history_sources_usage_and_database_items():
    source = SearchResult(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        document_name="Python.md",
        content="知识",
        score=0.9,
    )
    search = AsyncMock(return_value=[source])
    context, dependencies = runtime(search)
    model = ScriptedModel(
        responses=[
            model_message("先检索", 1, [call(1)]),
            model_message("最终回答", 2),
        ]
    )
    provider = LangChainProvider(model=model)
    events = [
        event
        async for event in provider.generate(
            [
                AgentMessage("user", "旧问题"),
                AgentMessage("assistant", "旧回答"),
                AgentMessage("user", "实际问题"),
            ],
            context,
            dependencies,
            instructions="本次规则",
        )
    ]
    assert [type(event) for event in events] == [
        AssistantMessageEvent,
        FunctionCallEvent,
        FunctionCallOutputEvent,
        AssistantMessageEvent,
        ModelCompletedEvent,
    ]
    assert [str(message.text) for message in model.inputs[0]] == [
        "本次规则",
        "旧问题",
        "旧回答",
        "实际问题",
    ]
    search.assert_awaited_once_with(query="Python", owner_id="actual-user")
    assert events[2].sources[0].content == "知识"
    assert events[2].call_id == events[1].call_id
    assert events[-1].total_tokens == 30
    assert events[-1].provider_response_id == "resp-2"
    rows = [
        _event_to_item(UUID(int=3), i, event) for i, event in enumerate(events[:-1])
    ]
    assert [row.item_type for row in rows] == [
        ItemType.MESSAGE,
        ItemType.FUNCTION_CALL,
        ItemType.FUNCTION_CALL_OUTPUT,
        ItemType.MESSAGE,
    ]
    assert rows[2].payload["sources"][0]["content"] == "知识"
    assert rows[2].payload["succeeded"] is True
    assert rows[3].provider_item_id is None
    assert provider.tool_definitions()[0]["name"] == "search_knowledge"


@pytest.mark.asyncio
async def test_streams_have_separate_keys_without_faking_provider_ids():
    context, dependencies = runtime()
    model = ScriptedModel(
        responses=[
            model_message("先查询", 1, [call(1)]),
            model_message("最终", 2),
        ]
    )
    events = [
        event
        async for event in LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
            stream=True,
        )
    ]
    deltas = [event for event in events if isinstance(event, AssistantTextDeltaEvent)]
    assert [event.delta for event in deltas] == ["先查询", "最终"]
    assert deltas[0].stream_key != deltas[1].stream_key
    assert [event.provider_response_id for event in deltas] == ["resp-1", "resp-2"]
    assert events[-1].total_tokens == 30


@pytest.mark.asyncio
async def test_parallel_tool_requests_do_not_share_session_concurrently():
    active = 0
    peak = 0

    async def search(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return []

    context, dependencies = runtime(search)
    model = ScriptedModel(
        responses=[
            model_message("", 1, [call(1), call(2)]),
            model_message("完成", 2),
        ]
    )
    events = [
        event
        async for event in LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
        )
    ]
    assert peak == 1
    assert (
        len([event for event in events if isinstance(event, FunctionCallOutputEvent)])
        == 2
    )


@pytest.mark.asyncio
async def test_tool_failure_is_safe_and_model_can_finish():
    context, dependencies = runtime(
        AsyncMock(side_effect=RuntimeError("secret database URL"))
    )
    model = ScriptedModel(
        responses=[
            model_message("", 1, [call(1)]),
            model_message("检索失败", 2),
        ]
    )
    events = [
        event
        async for event in LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
        )
    ]
    output = next(
        event for event in events if isinstance(event, FunctionCallOutputEvent)
    )
    assert output.succeeded is False
    assert output.sources == ()
    assert "secret" not in output.output
    assert isinstance(events[-1], ModelCompletedEvent)


@pytest.mark.asyncio
async def test_call_limit_prevents_tool_execution():
    context, dependencies = runtime()
    model = ScriptedModel(responses=[model_message("", 1, [call(1), call(2)])])
    with pytest.raises(ToolCallLimitExceeded):
        _ = [
            event
            async for event in LangChainProvider(
                model=model, max_tool_calls=1
            ).generate(
                [AgentMessage("user", "问题")],
                context,
                dependencies,
            )
        ]
    dependencies.search_service.search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,text",
    [
        ({"finish_reason": "length"}, "部分回答"),
        ({"status": "incomplete"}, "部分回答"),
        ({}, ""),
    ],
)
async def test_incomplete_or_empty_response_never_completes(metadata, text):
    context, dependencies = runtime()
    model = ScriptedModel(responses=[model_message(text, 1, **metadata)])
    events = []
    with pytest.raises(ModelResponseError):
        async for event in LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
        ):
            events.append(event)
    assert not any(isinstance(event, ModelCompletedEvent) for event in events)


@pytest.mark.asyncio
async def test_input_overflow_is_rejected_before_model_call():
    context, dependencies = runtime()
    model = ScriptedModel(responses=[model_message("回答", 1)])
    budget = ContextBudget(
        total_tokens=2000, output_tokens=200, safety_margin_tokens=100
    )
    with pytest.raises(ContextBudgetExceeded):
        _ = [
            event
            async for event in LangChainProvider(model=model, budget=budget).generate(
                [AgentMessage("user", "问题" * 2000)],
                context,
                dependencies,
            )
        ]
    assert model.inputs == []


@pytest.mark.asyncio
async def test_closing_consumer_closes_agent_stream():
    closed = False

    async def stream(*args, **kwargs):
        nonlocal closed
        try:
            yield (
                "messages",
                (AIMessageChunk(content="部分", id="lc-1"), {"langgraph_step": 1}),
            )
            await asyncio.Event().wait()
        finally:
            closed = True

    context, dependencies = runtime()
    model = ScriptedModel(responses=[])
    with patch(
        "interview_ai.agent.providers.langchain.create_agent",
        return_value=SimpleNamespace(astream=stream),
    ):
        events = LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
            stream=True,
        )
        assert isinstance(await anext(events), AssistantTextDeltaEvent)
        await events.aclose()
    assert closed


@pytest.mark.asyncio
async def test_oversized_retrieval_is_bounded_and_sources_match_model_input():
    source = SearchResult(chunk_id=UUID(int=1), content="资料" * 10000, score=0.9)
    context, dependencies = runtime(AsyncMock(return_value=[source]))
    model = ScriptedModel(
        responses=[
            model_message("", 1, [call(1), call(2)]),
            model_message("完成", 2),
        ]
    )
    events = [
        event
        async for event in LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
        )
    ]
    outputs = [event for event in events if isinstance(event, FunctionCallOutputEvent)]
    assert len(outputs) == 2
    for event in outputs:
        assert len(event.sources[0].content) < len(source.content)
        assert (
            event.sources[0].content
            == json.loads(event.output)["results"][0]["content"]
        )
        assert event.sources[0].end_line is None
    tool_inputs = [message for message in model.inputs[-1] if message.type == "tool"]
    assert [str(message.text) for message in tool_inputs] == [
        event.output for event in outputs
    ]


@pytest.mark.asyncio
async def test_budget_error_in_tool_aborts_instead_of_becoming_a_success():
    context, dependencies = runtime(
        AsyncMock(side_effect=ContextBudgetExceeded("too large"))
    )
    model = ScriptedModel(responses=[model_message("", 1, [call(1)])])
    events = []
    with pytest.raises(ContextBudgetExceeded):
        async for event in LangChainProvider(model=model).generate(
            [AgentMessage("user", "问题")],
            context,
            dependencies,
        ):
            events.append(event)
    assert any(isinstance(event, FunctionCallEvent) for event in events)
    assert not any(isinstance(event, ModelCompletedEvent) for event in events)


@pytest.mark.asyncio
async def test_provider_completes_through_conversation_service():
    from unittest.mock import MagicMock

    from sqlalchemy.ext.asyncio import AsyncSession

    from interview_ai.agent.context.builder import ContextBuilder
    from interview_ai.agent.repository import ConversationRepository
    from interview_ai.agent.services.conversation import ConversationService
    from interview_ai.db.models import Session, Turn

    context, dependencies = runtime()
    provider = LangChainProvider(
        model=ScriptedModel(
            responses=[
                model_message("", 1, [call(1)]),
                model_message("最终回答", 2),
            ]
        )
    )
    repository = AsyncMock(spec=ConversationRepository)
    repository.create_session.return_value = Session(
        id=UUID(int=4), owner_id=context.owner_id
    )
    repository.create_turn.return_value = Turn(
        id=UUID(int=5), session_id=UUID(int=4), sequence=0
    )
    repository.list_context_history.return_value = []
    service = ConversationService(
        MagicMock(spec=AsyncSession),
        repository,
        provider,
        ContextBuilder(summarizer=provider),
    )
    result = await service.chat(None, "实际问题", context, dependencies)
    assert result.answer == "最终回答"
    repository.complete_turn.assert_awaited_once()
    assert repository.complete_turn.await_args.kwargs["total_tokens"] == 30
    saved = repository.append_items.await_args_list[-1].args[0]
    assert [item.sequence for item in saved] == [1, 2, 3]
    assert [item.item_type for item in saved] == [
        ItemType.FUNCTION_CALL,
        ItemType.FUNCTION_CALL_OUTPUT,
        ItemType.MESSAGE,
    ]


@pytest.mark.asyncio
async def test_summary_and_lifecycle_delegate_to_existing_bounded_provider():
    model = ScriptedModel(responses=[])
    summary = SimpleNamespace(
        summarize_history=AsyncMock(return_value="摘要"), aclose=AsyncMock()
    )
    with patch(
        "interview_ai.agent.providers.langchain.OpenAIProvider", return_value=summary
    ):
        async with LangChainProvider(model=model) as provider:
            messages = [AgentMessage("user", "历史")]
            assert (
                await provider.summarize_history(messages, max_output_tokens=100)
                == "摘要"
            )
    summary.summarize_history.assert_awaited_once_with(messages, max_output_tokens=100)
    summary.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", [False, True], ids=["sequential", "concurrent"])
async def test_default_model_clients_survive_other_provider_closing(
    monkeypatch, overlap
):
    from interview_ai.config import config

    monkeypatch.setattr(config, "chat_api_key", "test-key")
    monkeypatch.setattr(config, "chat_api_host", "https://model.example.test/v1")
    first = LangChainProvider()
    second = None
    try:
        if overlap:
            second = LangChainProvider()
        await first.aclose()
        if second is None:
            second = LangChainProvider()
        assert first._model.root_async_client.is_closed()
        assert first._model.root_client.is_closed()
        assert not second._model.root_async_client.is_closed()
        assert not second._model.root_client.is_closed()
    finally:
        await first.aclose()
        if second is not None:
            await second.aclose()
    assert second._model.root_async_client.is_closed()
    assert second._model.root_client.is_closed()
