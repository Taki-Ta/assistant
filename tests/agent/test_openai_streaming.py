import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from interview_ai.agent.models import (
    AgentMessage,
    AssistantMessageEvent,
    AssistantTextDeltaEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
    ToolExecutionResult,
)
from interview_ai.agent.providers.openai import ModelResponseError, OpenAIProvider
from interview_ai.agent.runtime import AgentContext, ToolDependencies
from interview_ai.agent.tools.registry import ToolRegistry


class FakeStream:
    def __init__(self, *events):
        self.events = iter(events)
        self.consumed = 0
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = next(self.events, None)
        if event is None:
            raise StopAsyncIteration
        self.consumed += 1
        return event


def response(text="你好", output=None, response_id="response-1"):
    return SimpleNamespace(
        id=response_id,
        status="completed",
        output_text=text,
        output=output
        if output is not None
        else [SimpleNamespace(type="message", id="message-1")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def event(kind, **kwargs):
    return SimpleNamespace(type=kind, **kwargs)


def runtime(client, tools=()):
    return (
        OpenAIProvider(client=client, tools=ToolRegistry(tools)),
        AgentContext(owner_id="user-001"),
        ToolDependencies(search_service=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_text_deltas_arrive_before_completion_and_keep_message_identity():
    final = response()
    model_stream = FakeStream(
        event("response.created", response=final),
        event("response.output_text.delta", item_id="message-1", delta="你"),
        event("response.output_text.delta", item_id="message-1", delta="好"),
        event("response.completed", response=final),
    )
    client = AsyncMock()
    client.responses.create.return_value = model_stream
    provider, context, dependencies = runtime(client)
    generated = provider.generate(
        [AgentMessage(role="user", content="问候")], context, dependencies, stream=True
    )
    first = await anext(generated)
    assert isinstance(first, AssistantTextDeltaEvent)
    assert first.delta == "你"
    assert first.provider_item_id == "message-1"
    assert first.provider_response_id == "response-1"
    assert model_stream.consumed == 2
    assert not model_stream.closed
    remaining = [item async for item in generated]
    assert [type(item) for item in remaining] == [
        AssistantTextDeltaEvent,
        AssistantMessageEvent,
        ModelCompletedEvent,
    ]
    assert remaining[1].text == "你好"
    assert remaining[2].total_tokens == 15
    assert model_stream.closed
    assert client.responses.create.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_streaming_tool_loop_waits_for_complete_arguments_and_sums_usage():
    call = SimpleNamespace(
        type="function_call",
        id="item-call",
        call_id="call-1",
        name="search",
        arguments='{"query":"知识"}',
    )
    first_stream = FakeStream(
        event("response.function_call_arguments.delta", delta='{"query":'),
        event("response.completed", response=response("", [call], "tool-response")),
    )
    second_stream = FakeStream(
        event("response.output_text.delta", item_id="message-1", delta="你好"),
        event("response.completed", response=response()),
    )
    client = AsyncMock()
    client.responses.create.side_effect = [first_stream, second_stream]

    async def invoke(*args, **kwargs):
        assert first_stream.closed
        assert args[0] == '{"query":"知识"}'
        return ToolExecutionResult("[]")

    tool = SimpleNamespace(
        name="search",
        definition={"type": "function", "name": "search"},
        invoke=AsyncMock(side_effect=invoke),
    )
    provider, context, dependencies = runtime(client, [tool])
    events = [
        item
        async for item in provider.generate(
            [AgentMessage(role="user", content="查资料")],
            context,
            dependencies,
            stream=True,
        )
    ]
    assert [type(item) for item in events] == [
        FunctionCallEvent,
        FunctionCallOutputEvent,
        AssistantTextDeltaEvent,
        AssistantMessageEvent,
        ModelCompletedEvent,
    ]
    assert events[-1].input_tokens == 20
    assert events[-1].output_tokens == 10
    assert events[-1].total_tokens == 30
    assert first_stream.closed and second_stream.closed
    assert all(
        call.kwargs["stream"] for call in client.responses.create.await_args_list
    )
    second_input = client.responses.create.await_args.kwargs["input"]
    assert call in second_input
    assert {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "[]",
    } in second_input


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", ["response.failed", "response.incomplete", "error", None]
)
async def test_broken_stream_never_emits_complete_message_or_usage(terminal):
    model_stream = FakeStream(
        event("response.output_text.delta", item_id="partial", delta="部分"),
        *([event(terminal)] if terminal else []),
    )
    client = AsyncMock()
    client.responses.create.return_value = model_stream
    provider, context, dependencies = runtime(client)
    events = []
    with pytest.raises(ModelResponseError):
        async for item in provider.generate(
            [AgentMessage(role="user", content="问题")],
            context,
            dependencies,
            stream=True,
        ):
            events.append(item)
    assert [type(item) for item in events] == [AssistantTextDeltaEvent]
    assert model_stream.closed


@pytest.mark.asyncio
async def test_closing_generator_closes_upstream_stream():
    model_stream = FakeStream(
        event("response.output_text.delta", item_id="partial", delta="部分")
    )
    client = AsyncMock()
    client.responses.create.return_value = model_stream
    provider, context, dependencies = runtime(client)
    generated = provider.generate(
        [AgentMessage(role="user", content="问题")], context, dependencies, stream=True
    )
    await anext(generated)
    await generated.aclose()
    assert model_stream.closed


@pytest.mark.asyncio
async def test_cancelling_stream_consumer_closes_upstream_stream():
    started = asyncio.Event()

    class BlockingStream(FakeStream):
        async def __anext__(self):
            started.set()
            await asyncio.Event().wait()

    model_stream = BlockingStream()
    client = AsyncMock()
    client.responses.create.return_value = model_stream
    provider, context, dependencies = runtime(client)

    async def consume():
        return [
            item
            async for item in provider.generate(
                [AgentMessage(role="user", content="问题")],
                context,
                dependencies,
                stream=True,
            )
        ]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert model_stream.closed
