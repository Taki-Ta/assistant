import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from interview_ai.agent.models import (
    AgentMessage,
    AssistantMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
    RetrievedSource,
    ToolExecutionResult,
)
from interview_ai.agent.providers.openai import (
    OpenAIProvider,
    ToolCallLimitExceeded,
)
from interview_ai.agent.runtime import AgentContext, ToolDependencies
from interview_ai.agent.tools.registry import ToolRegistry
from interview_ai.agent.tools.search_knowledge import SearchKnowledgeTool
from interview_ai.indexing.models import SearchResult
from interview_ai.indexing.search_service import SearchService

CHUNK_ID = UUID("01900000-0000-7000-8000-000000000001")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000002")


def _function_call(call_id: str = "call-001") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"item-{call_id}",
        type="function_call",
        name="search_knowledge",
        arguments='{"query":"Python"}',
        call_id=call_id,
    )


def _final_response(text: str = "Python 是一种编程语言。") -> SimpleNamespace:
    return SimpleNamespace(
        id="response-final",
        output=[SimpleNamespace(id="message-final", type="message")],
        output_text=text,
    )


def _runtime() -> tuple[AgentContext, ToolDependencies]:
    return (
        AgentContext(owner_id="user-001"),
        ToolDependencies(search_service=AsyncMock(spec=SearchService)),
    )


def _source(*, score: float = 0.9) -> RetrievedSource:
    return RetrievedSource(
        chunk_id=CHUNK_ID,
        content="Python 是一种编程语言。",
        score=score,
        document_id=DOCUMENT_ID,
        document_name="Python.md",
        headings=("Python", "基础"),
        start_line=10,
        end_line=20,
    )


@pytest.mark.asyncio
async def test_generate_executes_tool_and_returns_final_answer() -> None:
    client = AsyncMock()
    client.responses.create = AsyncMock(
        side_effect=[
            SimpleNamespace(output=[_function_call()], output_text=""),
            _final_response(),
        ]
    )
    source = _source()
    tool = SimpleNamespace(
        name="search_knowledge",
        definition={
            "type": "function",
            "name": "search_knowledge",
            "parameters": {},
            "strict": False,
        },
        invoke=AsyncMock(
            return_value=ToolExecutionResult(
                output='[{"content":"Python"}]',
                sources=(source,),
            )
        ),
    )
    registry = ToolRegistry([tool])
    context, dependencies = _runtime()
    provider = OpenAIProvider(client=client, tools=registry)

    events = [
        event
        async for event in provider.generate(
            [AgentMessage(role="user", content="Python 是什么？")],
            context,
            dependencies,
        )
    ]

    assert [type(event) for event in events] == [
        FunctionCallEvent,
        FunctionCallOutputEvent,
        AssistantMessageEvent,
        ModelCompletedEvent,
    ]
    assert events[1].sources == (source,)
    assert events[2].text == "Python 是一种编程语言。"
    assert client.responses.create.await_count == 2
    assert client.responses.create.await_args_list[0].kwargs["input"] == [
        {"role": "user", "content": "Python 是什么？"}
    ]
    tool.invoke.assert_awaited_once_with('{"query":"Python"}', context, dependencies)
    second_input = client.responses.create.await_args_list[1].kwargs["input"]
    tool_outputs = [
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert tool_outputs == [
        {
            "type": "function_call_output",
            "call_id": "call-001",
            "output": '[{"content":"Python"}]',
        }
    ]


@pytest.mark.asyncio
async def test_generate_returns_tool_failure_to_model() -> None:
    client = AsyncMock()
    client.responses.create = AsyncMock(
        side_effect=[
            SimpleNamespace(output=[_function_call()], output_text=""),
            _final_response("知识库暂时不可用。"),
        ]
    )
    tool = SimpleNamespace(
        name="search_knowledge",
        definition={
            "type": "function",
            "name": "search_knowledge",
            "parameters": {},
            "strict": False,
        },
        invoke=AsyncMock(side_effect=RuntimeError("database unavailable")),
    )
    context, dependencies = _runtime()
    provider = OpenAIProvider(client=client, tools=ToolRegistry([tool]))

    events = [
        event
        async for event in provider.generate(
            [AgentMessage(role="user", content="查询知识库")],
            context,
            dependencies,
        )
    ]

    tool_event = next(
        event for event in events if isinstance(event, FunctionCallOutputEvent)
    )
    assert tool_event.succeeded is False
    assert tool_event.sources == ()
    assert (
        next(event.text for event in events if isinstance(event, AssistantMessageEvent))
        == "知识库暂时不可用。"
    )
    second_input = client.responses.create.await_args_list[1].kwargs["input"]
    tool_output = next(
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    error = json.loads(tool_output["output"])
    assert error == {
        "error": {
            "code": "tool_execution_failed",
            "message": "工具 search_knowledge 执行失败",
        }
    }


@pytest.mark.asyncio
async def test_generate_stops_after_configured_tool_call_limit() -> None:
    client = AsyncMock()
    client.responses.create = AsyncMock(
        side_effect=[
            SimpleNamespace(output=[_function_call("call-001")], output_text=""),
            SimpleNamespace(output=[_function_call("call-002")], output_text=""),
        ]
    )
    tool = SimpleNamespace(
        name="search_knowledge",
        definition={
            "type": "function",
            "name": "search_knowledge",
            "parameters": {},
            "strict": False,
        },
        invoke=AsyncMock(return_value=ToolExecutionResult(output="[]")),
    )
    context, dependencies = _runtime()
    provider = OpenAIProvider(
        client=client,
        tools=ToolRegistry([tool]),
        max_tool_calls=1,
    )

    with pytest.raises(ToolCallLimitExceeded, match="工具调用次数超过限制：1"):
        _ = [
            event
            async for event in provider.generate(
                [AgentMessage(role="user", content="不断搜索")],
                context,
                dependencies,
            )
        ]

    assert tool.invoke.await_count == 1


@pytest.mark.asyncio
async def test_registry_rejects_unknown_tool() -> None:
    context, dependencies = _runtime()

    with pytest.raises(LookupError, match="未知工具：missing"):
        await ToolRegistry().invoke("missing", "{}", context, dependencies)


@pytest.mark.asyncio
async def test_generate_emits_sources_for_each_tool_call() -> None:
    client = AsyncMock()
    client.responses.create = AsyncMock(
        side_effect=[
            SimpleNamespace(
                output=[
                    _function_call("call-001"),
                    _function_call("call-002"),
                ],
                output_text="",
            ),
            _final_response(),
        ]
    )
    first_source = _source(score=0.8)
    latest_source = _source(score=0.9)
    tool = SimpleNamespace(
        name="search_knowledge",
        definition={
            "type": "function",
            "name": "search_knowledge",
            "parameters": {},
            "strict": False,
        },
        invoke=AsyncMock(
            side_effect=[
                ToolExecutionResult(output="[]", sources=(first_source,)),
                ToolExecutionResult(output="[]", sources=(latest_source,)),
            ]
        ),
    )
    context, dependencies = _runtime()
    provider = OpenAIProvider(client=client, tools=ToolRegistry([tool]))

    events = [
        event
        async for event in provider.generate(
            [AgentMessage(role="user", content="搜索两次")],
            context,
            dependencies,
        )
    ]
    output_events = [
        event for event in events if isinstance(event, FunctionCallOutputEvent)
    ]

    assert [event.sources for event in output_events] == [
        (first_source,),
        (latest_source,),
    ]


@pytest.mark.asyncio
async def test_generate_converts_conversation_history_to_openai_input() -> None:
    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=_final_response("继续回答"))
    context, dependencies = _runtime()
    provider = OpenAIProvider(client=client, tools=ToolRegistry())

    _ = [
        event
        async for event in provider.generate(
            [
                AgentMessage(role="user", content="第一问"),
                AgentMessage(role="assistant", content="第一答"),
                AgentMessage(role="user", content="继续问"),
            ],
            context,
            dependencies,
        )
    ]

    assert client.responses.create.await_args.kwargs["input"] == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "继续问"},
    ]


@pytest.mark.asyncio
async def test_search_knowledge_returns_output_and_retrieved_sources() -> None:
    context, dependencies = _runtime()
    search_result = SearchResult(
        chunk_id=CHUNK_ID,
        content="Python 是一种编程语言。",
        score=0.9,
        document_id=DOCUMENT_ID,
        document_name="Python.md",
        headings=("Python", "基础"),
        start_line=10,
        end_line=20,
    )
    dependencies.search_service.search.return_value = [search_result]

    result = await SearchKnowledgeTool().invoke(
        '{"query":"Python"}',
        context,
        dependencies,
    )

    dependencies.search_service.search.assert_awaited_once_with(
        query="Python",
        owner_id="user-001",
    )
    assert json.loads(result.output) == [
        {
            "chunk_id": str(CHUNK_ID),
            "content": "Python 是一种编程语言。",
            "score": 0.9,
            "document_id": str(DOCUMENT_ID),
            "document_name": "Python.md",
            "headings": ["Python", "基础"],
            "start_line": 10,
            "end_line": 20,
        }
    ]
    assert result.sources == (_source(),)
