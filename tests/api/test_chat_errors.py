from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from openai import OpenAIError

from interview_ai.agent.context.budget import (
    CompactionError,
    ContextBudgetExceeded,
    InputTooLarge,
)
from interview_ai.agent.providers.openai import ModelResponseError
from interview_ai.api.routes.ai import chat
from interview_ai.api.schemas import ChatRequest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type, status_code",
    [
        (InputTooLarge, 413),
        (ContextBudgetExceeded, 422),
        (CompactionError, 502),
        (ModelResponseError, 502),
        (OpenAIError, 502),
    ],
)
async def test_chat_returns_safe_errors(error_type, status_code):
    service = AsyncMock()
    service.chat.side_effect = error_type("private-key-or-endpoint")
    with pytest.raises(HTTPException) as error:
        await chat(ChatRequest(message="问题"), service, None, None)
    assert error.value.status_code == status_code
    assert "private-key-or-endpoint" not in error.value.detail
