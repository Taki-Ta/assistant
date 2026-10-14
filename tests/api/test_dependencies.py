from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zhiyin.api import dependencies
from zhiyin.api.dependencies import (
    ConversationDatabaseSession,
    DatabaseSession,
)
from zhiyin.db import database


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "langchain"])
async def test_conversation_provider_selection_preserves_context_builder(
    monkeypatch, provider_name
):
    from zhiyin.agent.providers import langchain

    provider = MagicMock()
    provider_context = MagicMock()
    provider_context.__aenter__.return_value = provider
    factory = MagicMock(return_value=provider_context)
    monkeypatch.setattr(dependencies.config, "chat_provider", provider_name)
    if provider_name == "openai":
        monkeypatch.setattr(dependencies, "OpenAIProvider", factory)
    else:
        monkeypatch.setattr(langchain, "LangChainProvider", factory)
    events = dependencies.get_conversation_service(MagicMock())
    service = await anext(events)
    assert service._provider is provider
    assert service._context_builder.summarizer is provider
    assert factory.call_args.kwargs["budget"] == service._context_builder.budget
    await events.aclose()
    provider_context.__aexit__.assert_awaited_once()


def test_conversation_and_request_dependencies_use_different_sessions(
    monkeypatch,
) -> None:
    created_sessions: list[object] = []

    class SessionContext:
        async def __aenter__(self) -> object:
            session = object()
            created_sessions.append(session)
            return session

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(database, "session_factory", SessionContext)

    app = FastAPI()

    @app.get("/sessions")
    async def sessions(
        db: DatabaseSession,
        conversation_db: ConversationDatabaseSession,
    ) -> dict[str, bool]:
        return {"same": db is conversation_db}

    response = TestClient(app).get("/sessions")

    assert response.status_code == 200
    assert response.json() == {"same": False}
    assert len(created_sessions) == 2
