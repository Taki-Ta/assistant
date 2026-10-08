import json
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock
from uuid import UUID

import pytest

from interview_ai.agent.context.budget import ContextBudget, ContextBudgetExceeded
from interview_ai.agent.context.estimator import estimate_request
from interview_ai.agent.context.tool_output import ToolOutputBudget
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
    ModelResponseError,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_status, answer",
    [
        ("incomplete", "部分回答"),
        ("failed", ""),
        ("completed", "  "),
    ],
)
async def test_unfinished_or_empty_response_never_emits_completed(
    response_status, answer
):
    client = AsyncMock()
    response = _final_response(answer)
    response.status = response_status
    client.responses.create.return_value = response
    provider = OpenAIProvider(client=client, tools=ToolRegistry())
    context, dependencies = _runtime()
    events = []
    with pytest.raises(ModelResponseError):
        async for event in provider.generate(
            [AgentMessage(role="user", content="问题")],
            context,
            dependencies,
        ):
            events.append(event)
    assert events == []


@pytest.mark.asyncio
async def test_incomplete_summary_is_rejected():
    client = AsyncMock()
    client.responses.create.return_value = SimpleNamespace(
        status="incomplete", output_text="部分摘要"
    )
    provider = OpenAIProvider(client=client, tools=ToolRegistry())
    with pytest.raises(ModelResponseError):
        await provider.summarize_history(
            [AgentMessage(role="user", content="历史")], max_output_tokens=100
        )


@pytest.mark.asyncio
async def test_summary_is_separate_request_without_tools():
    client = AsyncMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="历史摘要", status="completed"
    )
    provider = OpenAIProvider(client=client, tools=ToolRegistry())
    result = await provider.summarize_history(
        [AgentMessage(role="user", content="历史问题")],
        max_output_tokens=100,
    )
    assert result == "历史摘要"
    kwargs = client.responses.create.await_args.kwargs
    assert "tools" not in kwargs
    assert kwargs["max_output_tokens"] == 100


@pytest.mark.asyncio
async def test_tool_loop_rejects_overflow_before_second_model_request():
    client = AsyncMock()
    client.responses.create.return_value = SimpleNamespace(
        output=[_function_call()], output_text=""
    )
    tool = SimpleNamespace(
        name="search_knowledge",
        definition={"type": "function", "name": "search_knowledge"},
        invoke=AsyncMock(return_value=ToolExecutionResult(output="结果" * 1000)),
    )
    provider = OpenAIProvider(
        client=client,
        tools=ToolRegistry([tool]),
        budget=ContextBudget(
            total_tokens=1000, output_tokens=100, safety_margin_tokens=100
        ),
    )
    context, dependencies = _runtime()
    with pytest.raises(ContextBudgetExceeded, match="工具结果"):
        _ = [
            event
            async for event in provider.generate(
                [AgentMessage(role="user", content="查询")],
                context,
                dependencies,
            )
        ]
    assert client.responses.create.await_count == 1
    assert client.responses.create.await_args.kwargs["max_output_tokens"] == 100


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
    tool.invoke.assert_awaited_once_with(
        '{"query":"Python"}', context, dependencies, output_budget=ANY
    )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_bounded_search_results_allow_multiple_calls_to_finish(parallel):
    client = AsyncMock()
    calls = [_function_call("first"), _function_call("second")]
    responses = (
        [SimpleNamespace(output=calls, output_text="")]
        if parallel
        else [SimpleNamespace(output=[call], output_text="") for call in calls]
    )
    client.responses.create.side_effect = [*responses, _final_response("完成")]
    budget = ContextBudget(
        total_tokens=8000,
        output_tokens=800,
        safety_margin_tokens=400,
        tool_output_tokens=2000,
    )
    provider = OpenAIProvider(
        client=client, tools=ToolRegistry([SearchKnowledgeTool()]), budget=budget
    )
    context, dependencies = _runtime()
    dependencies.search_service.search.return_value = [
        SearchResult(
            chunk_id=CHUNK_ID,
            content='资料"\n' * 5000,
            score=0.9,
            document_name="知识库",
            start_line=1,
            end_line=5000,
        )
    ]
    messages = [AgentMessage(role="user", content="问题" * 1800)]
    events = [
        event async for event in provider.generate(messages, context, dependencies)
    ]
    outputs = [event for event in events if isinstance(event, FunctionCallOutputEvent)]
    assert len(outputs) == 2
    assert events[-2].text == "完成"
    for event in outputs:
        payload = json.loads(event.output)
        assert payload["truncated"] is True
        assert payload["results"][0]["content"] == event.sources[0].content
        assert event.sources[0].end_line is None
        assert len(event.sources[0].content) < len(
            dependencies.search_service.search.return_value[0].content
        )
    for call in client.responses.create.await_args_list:
        kwargs = call.kwargs
        assert (
            estimate_request(kwargs["instructions"], kwargs["input"], kwargs["tools"])
            <= budget.input_tokens
        )
    final_input = client.responses.create.await_args.kwargs["input"]
    sent_outputs = [
        item
        for item in final_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in sent_outputs] == ["first", "second"]
    assert [item["output"] for item in sent_outputs] == [
        event.output for event in outputs
    ]
    assert (
        estimate_request(None, final_input, provider.tool_definitions())
        > budget.initial_input_tokens
    )


@pytest.mark.asyncio
async def test_search_budget_keeps_complete_highest_ranked_source():
    context, dependencies = _runtime()
    dependencies.search_service.search.return_value = [
        SearchResult(chunk_id=DOCUMENT_ID, content="低相关内容" * 30, score=0.2),
        SearchResult(chunk_id=CHUNK_ID, content="高相关内容" * 30, score=0.9),
    ]
    output_budget = ToolOutputBudget(350, "search")
    result = await SearchKnowledgeTool().invoke(
        '{"query":"Python"}', context, dependencies, output_budget=output_budget
    )
    assert output_budget.fits(result.output)
    assert [source.chunk_id for source in result.sources] == [CHUNK_ID]
    assert result.sources[0].content == "高相关内容" * 30
    payload = json.loads(result.output)
    assert payload["truncated"] is True
    assert [item["chunk_id"] for item in payload["results"]] == [str(CHUNK_ID)]


@pytest.mark.asyncio
async def test_search_does_not_report_no_hits_when_metadata_cannot_fit():
    context, dependencies = _runtime()
    dependencies.search_service.search.return_value = [
        SearchResult(
            chunk_id=CHUNK_ID, content="资料", score=0.9, document_name="标题" * 1000
        )
    ]
    with pytest.raises(ContextBudgetExceeded, match="有效检索片段"):
        await SearchKnowledgeTool().invoke(
            '{"query":"Python"}',
            context,
            dependencies,
            output_budget=ToolOutputBudget(100, "search"),
        )


@pytest.mark.asyncio
async def test_tool_calls_fail_before_execution_when_minimum_outputs_cannot_fit():
    client = AsyncMock()
    client.responses.create.return_value = SimpleNamespace(
        output=[_function_call("x" * 1000)], output_text=""
    )
    tool = SimpleNamespace(
        name="search_knowledge",
        definition={"name": "search_knowledge"},
        invoke=AsyncMock(),
    )
    provider = OpenAIProvider(
        client=client,
        tools=ToolRegistry([tool]),
        budget=ContextBudget(
            total_tokens=600, output_tokens=100, safety_margin_tokens=50
        ),
    )
    context, dependencies = _runtime()
    with pytest.raises(ContextBudgetExceeded, match="工具结果"):
        _ = [
            event
            async for event in provider.generate(
                [AgentMessage(role="user", content="检索")], context, dependencies
            )
        ]
    tool.invoke.assert_not_awaited()
    assert client.responses.create.await_count == 1
