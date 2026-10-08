import asyncio
import json
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from interview_ai.agent.models import (
    AssistantTextDeltaEvent,
    ChatCompletedEvent,
    ChatStartedEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)
from interview_ai.agent.providers.openai import ModelResponseError
from interview_ai.agent.services import SessionNotFoundError
from interview_ai.api.dependencies import (
    get_agent_context,
    get_conversation_service,
    get_tool_dependencies,
)
from interview_ai.api.routes.ai import chat_stream, router
from interview_ai.api.schemas import ChatRequest

SESSION_ID = UUID("01900000-0000-7000-8000-000000000010")
TURN_ID = UUID("01900000-0000-7000-8000-000000000011")


def client_for(service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: service
    app.dependency_overrides[get_agent_context] = lambda: None
    app.dependency_overrides[get_tool_dependencies] = lambda: None
    return TestClient(app)


def decode_sse(text):
    return [
        (
            frame.splitlines()[0].removeprefix("event: "),
            json.loads(frame.splitlines()[1].removeprefix("data: ")),
        )
        for frame in text.strip().split("\n\n")
    ]


@pytest.mark.parametrize("tool_succeeded", [True, False])
def test_stream_endpoint_serializes_events_and_keeps_dependencies_alive(tool_succeeded):
    service = MagicMock()
    alive = False
    closed = False

    async def stream(*args):
        nonlocal closed
        try:
            assert alive
            yield ChatStartedEvent(session_id=SESSION_ID, turn_id=TURN_ID)
            assert alive
            yield FunctionCallEvent(
                call_id="call-1", tool_name="search", arguments={"query": "你好"}
            )
            yield FunctionCallOutputEvent(
                call_id="call-1",
                tool_name="search",
                output='{"结果":"资料\\n内容"}',
                succeeded=tool_succeeded,
            )
            yield AssistantTextDeltaEvent(delta='文字\n"换行"')
            yield ChatCompletedEvent(
                session_id=SESSION_ID, turn_id=TURN_ID, answer='文字\n"换行"'
            )
        finally:
            closed = True

    service.chat_stream.side_effect = stream
    client = client_for(service)

    async def dependency():
        nonlocal alive
        alive = True
        try:
            yield service
        finally:
            alive = False

    client.app.dependency_overrides[get_conversation_service] = dependency
    response = client.post("/api/v1/chat/stream", json={"message": "你好"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    frames = decode_sse(response.text)
    assert [name for name, _ in frames] == [
        "started",
        "function_call",
        "function_call_output",
        "text_delta",
        "completed",
    ]
    assert frames[0][1]["session_id"] == str(SESSION_ID)
    assert frames[1][1]["arguments"] == {"query": "你好"}
    assert frames[2][1]["call_id"] == frames[1][1]["call_id"]
    assert frames[2][1]["succeeded"] is tool_succeeded
    assert json.loads(frames[2][1]["output"]) == {"结果": "资料\n内容"}
    assert frames[3][1]["delta"] == '文字\n"换行"'
    assert closed and not alive


def test_stream_maps_missing_session_to_http_404_before_headers():
    service = MagicMock()

    async def stream(*args):
        raise SessionNotFoundError("private detail")
        yield

    service.chat_stream.side_effect = stream
    response = client_for(service).post(
        "/api/v1/chat/stream", json={"message": "你好", "session_id": str(SESSION_ID)}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


@pytest.mark.parametrize(
    "error", [ModelResponseError("private-key"), RuntimeError("private-key")]
)
def test_stream_errors_are_safe_sse_events_without_completed(error):
    service = MagicMock()

    async def stream(*args):
        yield ChatStartedEvent(session_id=SESSION_ID, turn_id=TURN_ID)
        yield AssistantTextDeltaEvent(delta="部分")
        raise error

    service.chat_stream.side_effect = stream
    response = client_for(service).post("/api/v1/chat/stream", json={"message": "问题"})
    assert response.status_code == 200
    frames = decode_sse(response.text)
    assert [name for name, _ in frames] == ["started", "text_delta", "error"]
    assert "private-key" not in response.text
    assert frames[-1][1]["turn_id"] == str(TURN_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ["http.response.start", "http.response.body"])
async def test_response_send_failure_closes_primed_generator(fail_at):
    service = MagicMock()
    closed = False

    async def stream(*args):
        nonlocal closed
        try:
            yield ChatStartedEvent(session_id=SESSION_ID, turn_id=TURN_ID)
            yield AssistantTextDeltaEvent(delta="文字")
        finally:
            closed = True

    service.chat_stream.side_effect = stream
    response = await chat_stream(ChatRequest(message="问题"), service, None, None)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == fail_at:
            raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert closed


@pytest.mark.asyncio
async def test_heartbeats_keep_waiting_on_same_event_without_cancelling_work(
    monkeypatch,
):
    monkeypatch.setattr(
        "interview_ai.api.routes.ai.SSE_HEARTBEAT_INTERVAL_SECONDS", 0.01
    )
    service = MagicMock()
    release = asyncio.Event()
    readers = []
    closed = False

    async def stream(*args):
        nonlocal closed
        try:
            yield ChatStartedEvent(session_id=SESSION_ID, turn_id=TURN_ID)
            readers.append(asyncio.current_task())
            await release.wait()
            yield AssistantTextDeltaEvent(delta="回答")
            yield ChatCompletedEvent(
                session_id=SESSION_ID, turn_id=TURN_ID, answer="回答"
            )
        finally:
            closed = True

    service.chat_stream.side_effect = stream
    response = await chat_stream(ChatRequest(message="问题"), service, None, None)
    body = response.body_iterator
    try:
        assert (await anext(body)).startswith("event: started")
        assert await asyncio.wait_for(anext(body), 1) == ": ping\n\n"
        assert await asyncio.wait_for(anext(body), 1) == ": ping\n\n"
        assert len(readers) == 1
        assert not readers[0].done()
        assert not closed
        release.set()
        delta = await asyncio.wait_for(anext(body), 1)
        assert decode_sse(delta)[0][0] == "text_delta"
        completed = await asyncio.wait_for(anext(body), 1)
        assert decode_sse(completed)[0][0] == "completed"
        with pytest.raises(StopAsyncIteration):
            await anext(body)
    finally:
        await body.aclose()
        await response._events.aclose()
    assert closed and readers[0].done()


@pytest.mark.asyncio
async def test_disconnect_during_heartbeat_cancels_reader_and_closes_session(
    monkeypatch,
):
    monkeypatch.setattr(
        "interview_ai.api.routes.ai.SSE_HEARTBEAT_INTERVAL_SECONDS", 0.01
    )
    service = MagicMock()
    readers = []
    closed = False

    async def stream(*args):
        nonlocal closed
        try:
            yield ChatStartedEvent(session_id=SESSION_ID, turn_id=TURN_ID)
            readers.append(asyncio.current_task())
            await asyncio.Event().wait()
        finally:
            closed = True

    service.chat_stream.side_effect = stream
    response = await chat_stream(ChatRequest(message="问题"), service, None, None)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message.get("body") == b": ping\n\n":
            raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await asyncio.wait_for(
            response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send),
            1,
        )
    assert closed
    assert len(readers) == 1
    assert readers[0].done() and readers[0].cancelled()
