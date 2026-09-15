import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from interview_ai.agent.models import RetrievedSource, ToolExecutionResult
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
        type="function_call",
        name="search_knowledge",
        arguments='{"query":"Python"}',
        call_id=call_id,
    )


def _final_response(text: str = "Python 是一种编程语言。") -> SimpleNamespace:
    return SimpleNamespace(
        output=[SimpleNamespace(type="message")],
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

    result = await provider.generate("Python 是什么？", context, dependencies)

    assert result.answer == "Python 是一种编程语言。"
    assert result.sources == (source,)
    assert client.responses.create.await_count == 2
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

    result = await provider.generate("查询知识库", context, dependencies)

    assert result.answer == "知识库暂时不可用。"
    assert result.sources == ()
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
        await provider.generate("不断搜索", context, dependencies)

    assert tool.invoke.await_count == 1


@pytest.mark.asyncio
async def test_registry_rejects_unknown_tool() -> None:
    context, dependencies = _runtime()

    with pytest.raises(LookupError, match="未知工具：missing"):
        await ToolRegistry().invoke("missing", "{}", context, dependencies)


@pytest.mark.asyncio
async def test_generate_deduplicates_sources_by_chunk_id() -> None:
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

    result = await provider.generate("搜索两次", context, dependencies)

    assert result.sources == (latest_source,)


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
        limit=5,
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
