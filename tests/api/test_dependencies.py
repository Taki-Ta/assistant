from fastapi import FastAPI
from fastapi.testclient import TestClient

from interview_ai.api.dependencies import (
    ConversationDatabaseSession,
    DatabaseSession,
)
from interview_ai.db import database


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
