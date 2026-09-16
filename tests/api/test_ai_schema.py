from uuid import UUID

import pytest
from pydantic import ValidationError

from interview_ai.api.schemas import ChatRequest, ChatResponse

SESSION_ID = UUID("01900000-0000-7000-8000-000000000010")


def test_chat_request_parses_session_id() -> None:
    request = ChatRequest(message="你好", session_id=str(SESSION_ID))

    assert request.session_id == SESSION_ID


def test_chat_request_rejects_invalid_session_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="你好", session_id="not-a-uuid")


def test_chat_response_exposes_retrieved_sources() -> None:
    response = ChatResponse(
        answer="回答",
        session_id=SESSION_ID,
        retrieved_sources=(),
    )

    assert response.model_dump()["retrieved_sources"] == ()
    assert "sources" not in response.model_dump()
