from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from uuid6 import uuid7

from zhiyin.agent.runtime import AgentContext
from zhiyin.agent.services import SessionNotFoundError
from zhiyin.agent.services.session import (
    InvalidCursorError,
    SessionPage,
    SessionService,
    TurnPage,
    TurnRecord,
)
from zhiyin.api.dependencies import get_agent_context, get_session_service
from zhiyin.api.routes.session import router
from zhiyin.db.models import (
    ConversationItem,
    ItemRole,
    ItemType,
    Session,
    Turn,
    TurnStatus,
)


def client_for(service, *, authenticated=True):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session_service] = lambda: service
    if authenticated:
        app.dependency_overrides[get_agent_context] = lambda: AgentContext(
            owner_id="owner"
        )
    return TestClient(app)


def test_session_list_exposes_only_display_fields_and_forwards_cursor():
    service = AsyncMock(spec=SessionService)
    session = Session(owner_id="private-owner", title="知识库")
    service.list_sessions.return_value = SessionPage([session], "next")
    response = client_for(service).get("/api/v1/sessions?limit=5&cursor=previous")
    assert response.status_code == 200
    data = response.json()
    assert data["next_cursor"] == "next"
    assert set(data["items"][0]) == {"id", "title", "created_at", "updated_at"}
    assert data["items"][0]["title"] == "知识库"
    service.list_sessions.assert_awaited_once_with("owner", limit=5, cursor="previous")


def test_session_detail():
    service = AsyncMock(spec=SessionService)
    session = Session(owner_id="owner", title="问题")
    service.get_session.return_value = session
    response = client_for(service).get(f"/api/v1/sessions/{session.id}")
    assert response.status_code == 200
    assert response.json()["id"] == str(session.id)
    assert "owner_id" not in response.json()
    service.get_session.assert_awaited_once_with(session.id, "owner")


def test_turn_history_serializes_tools_sources_and_safe_failure():
    service = AsyncMock(spec=SessionService)
    session_id = uuid7()
    turn = Turn(
        session_id=session_id,
        sequence=2,
        status=TurnStatus.FAILED,
        error_message="secret-api-key",
        model="private-model",
    )
    common = {"turn_id": turn.id}
    source = {
        "chunk_id": str(uuid7()),
        "content": "知识库资料",
        "score": 0.9,
        "document_name": "文档",
        "headings": ["章节"],
        "start_line": 1,
        "end_line": 2,
    }
    items = [
        ConversationItem(
            **common,
            sequence=0,
            item_type=ItemType.MESSAGE,
            role=ItemRole.USER,
            text_content="问题",
        ),
        ConversationItem(
            **common,
            sequence=1,
            item_type=ItemType.FUNCTION_CALL,
            call_id="call",
            tool_name="search_knowledge",
            payload={
                "arguments": {"query": "知识库"},
                "provider_response_id": "private",
            },
        ),
        ConversationItem(
            **common,
            sequence=2,
            item_type=ItemType.FUNCTION_CALL_OUTPUT,
            call_id="call",
            tool_name="search_knowledge",
            text_content="结果",
            payload={"succeeded": True, "sources": [source]},
        ),
        ConversationItem(
            **common,
            sequence=3,
            item_type=ItemType.MESSAGE,
            role=ItemRole.ASSISTANT,
            text_content="回答",
        ),
    ]
    service.list_turns.return_value = TurnPage([TurnRecord(turn, items)], 2)
    response = client_for(service).get(
        f"/api/v1/sessions/{session_id}/turns?limit=3&cursor=5"
    )
    assert response.status_code == 200
    service.list_turns.assert_awaited_once_with(session_id, "owner", limit=3, cursor=5)
    data = response.json()
    assert data["next_cursor"] == 2
    record = data["items"][0]
    assert record["status"] == "failed"
    assert record["error_message"] == "本轮回答未正常完成"
    assert "secret-api-key" not in response.text
    assert "private-model" not in response.text
    assert "provider_response_id" not in response.text
    assert [item["type"] for item in record["items"]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert record["items"][1]["arguments"] == {"query": "知识库"}
    assert record["items"][2]["call_id"] == record["items"][1]["call_id"]
    assert record["items"][2]["sources"][0]["content"] == "知识库资料"


@pytest.mark.parametrize(
    "status, message",
    [
        (TurnStatus.IN_PROGRESS, None),
        (TurnStatus.COMPLETED, None),
        (TurnStatus.CANCELLED, "本轮回答已停止"),
    ],
)
def test_turn_statuses_remain_visible(status, message):
    service = AsyncMock(spec=SessionService)
    turn = Turn(session_id=uuid7(), sequence=0, status=status)
    service.list_turns.return_value = TurnPage([TurnRecord(turn, [])], None)
    data = client_for(service).get(f"/api/v1/sessions/{turn.session_id}/turns").json()
    assert data["items"][0]["status"] == status
    assert data["items"][0]["error_message"] == message


def test_delete_returns_empty_204():
    service = AsyncMock(spec=SessionService)
    session_id = uuid7()
    response = client_for(service).delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 204
    assert response.content == b""
    service.delete_session.assert_awaited_once_with(session_id, "owner")


@pytest.mark.parametrize(
    "method,suffix,operation",
    [
        ("get", "", "get_session"),
        ("get", "/turns", "list_turns"),
        ("delete", "", "delete_session"),
    ],
)
def test_missing_deleted_and_other_users_sessions_return_same_404(
    method, suffix, operation
):
    service = AsyncMock(spec=SessionService)
    getattr(service, operation).side_effect = SessionNotFoundError("private-session")
    response = getattr(client_for(service), method)(
        f"/api/v1/sessions/{uuid7()}{suffix}"
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/sessions"),
        ("get", f"/api/v1/sessions/{uuid7()}"),
        ("get", f"/api/v1/sessions/{uuid7()}/turns"),
        ("delete", f"/api/v1/sessions/{uuid7()}"),
    ],
)
def test_session_routes_require_authentication(method, path):
    service = AsyncMock(spec=SessionService)
    response = getattr(client_for(service, authenticated=False), method)(path)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert service.mock_calls == []


@pytest.mark.parametrize(
    "suffix",
    [
        "?limit=0",
        "?limit=101",
        "?limit=invalid",
        "/not-a-uuid",
        f"/{uuid7()}/turns?cursor=-1",
        f"/{uuid7()}/turns?cursor=bad",
    ],
)
def test_invalid_parameters_are_rejected_before_service_call(suffix):
    service = AsyncMock(spec=SessionService)
    response = client_for(service).get(f"/api/v1/sessions{suffix}")
    assert response.status_code == 422
    assert service.mock_calls == []


def test_malformed_session_cursor_returns_400():
    service = AsyncMock(spec=SessionService)
    service.list_sessions.side_effect = InvalidCursorError("private-details")
    response = client_for(service).get("/api/v1/sessions?cursor=bad")
    assert response.status_code == 400
    assert response.json() == {"detail": "无效的会话分页游标"}


def test_turn_cursor_outside_database_integer_range_is_rejected():
    service = AsyncMock(spec=SessionService)
    response = client_for(service).get(
        f"/api/v1/sessions/{uuid7()}/turns?cursor=2147483648"
    )
    assert response.status_code == 422
    service.list_turns.assert_not_awaited()


def test_token_without_subject_is_rejected():
    from zhiyin.api.auth import verify_bearer

    service = AsyncMock(spec=SessionService)
    client = client_for(service, authenticated=False)
    client.app.dependency_overrides[verify_bearer] = lambda: {"sub": ""}
    response = client.get("/api/v1/sessions")
    assert response.status_code == 401
    service.list_sessions.assert_not_awaited()
