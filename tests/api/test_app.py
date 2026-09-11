from fastapi.testclient import TestClient

from interview_ai.api.app import create_app
from interview_ai.config import config


def test_public_system_endpoints_are_available():
    client = TestClient(create_app())

    assert client.get("/").json() == {"message": "hello world"}
    assert client.get("/health").json() == {"status": "ok"}


def test_api_documentation_is_disabled_outside_debug_mode(monkeypatch):
    monkeypatch.setattr(config, "is_debug", False)
    client = TestClient(create_app())

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
