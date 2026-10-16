from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI, OpenAI
from starlette.requests import Request

from zhiyin.api.app import create_app
from zhiyin.api.dependencies import (
    get_embedding_provider,
    get_index_service,
    get_search_service,
)
from zhiyin.config import config
from zhiyin.indexing.embeddings import factory
from zhiyin.indexing.embeddings.langchain import LangChainEmbeddingProvider
from zhiyin.indexing.embeddings.openai import OpenAIEmbeddingProvider


@pytest.fixture
def embedding_clients(monkeypatch):
    async_client = AsyncOpenAI(api_key="test-key")
    sync_client = OpenAI(api_key="test-key")
    sync_factory = Mock(return_value=sync_client)
    monkeypatch.setattr(factory, "AsyncOpenAI", lambda **kwargs: async_client)
    monkeypatch.setattr(factory, "OpenAI", sync_factory)
    monkeypatch.setattr(config, "api_key", "test-key")
    monkeypatch.setattr(config, "embedding_model_name", "text-embedding-3-small")
    monkeypatch.setattr(config, "embedding_batch_size", 2)
    monkeypatch.setattr(config, "dimensions", 2)
    yield async_client, sync_client, sync_factory
    sync_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_during_serving", [False, True])
@pytest.mark.parametrize("provider_name", ["openai", "langchain"])
async def test_lifespan_initializes_selected_provider_and_closes_clients(
    embedding_clients, fail_during_serving, provider_name, monkeypatch
):
    async_client, sync_client, sync_factory = embedding_clients
    monkeypatch.setattr(config, "embedding_provider", provider_name)
    app = create_app()

    async def serve():
        async with app.router.lifespan_context(app):
            provider = app.state.embedding_provider
            assert provider.model_name == config.embedding_model_name
            if provider_name == "langchain":
                assert isinstance(provider, LangChainEmbeddingProvider)
                assert provider.client.dimensions == config.dimensions
                assert provider.client.chunk_size == config.embedding_batch_size
                assert provider.client.check_embedding_ctx_length is False
                assert provider.client.async_client is async_client.embeddings
                assert provider.client.client is sync_client.embeddings
            else:
                assert isinstance(provider, OpenAIEmbeddingProvider)
                sync_factory.assert_not_called()
            request = Request({"type": "http", "app": app})
            selected = get_embedding_provider(request)
            assert selected is provider
            assert get_index_service(Mock(), selected).embedding_provider is provider
            assert get_search_service(Mock(), selected).embedding_provider is provider
            assert not async_client.is_closed()
            assert not sync_client.is_closed()
            if fail_during_serving:
                raise RuntimeError("serving failed")

    if fail_during_serving:
        with pytest.raises(RuntimeError, match="serving failed"):
            await serve()
    else:
        await serve()

    assert async_client.is_closed()
    assert sync_client.is_closed() == (provider_name == "langchain")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "langchain"])
async def test_lifespan_closes_clients_when_initialization_fails(
    embedding_clients, monkeypatch, provider_name
):
    async_client, sync_client, sync_factory = embedding_clients
    monkeypatch.setattr(config, "embedding_provider", provider_name)

    def fail_initialization(**kwargs):
        raise ValueError("invalid embeddings configuration")

    target = (
        "OpenAIEmbeddings"
        if provider_name == "langchain"
        else "OpenAIEmbeddingProvider"
    )
    monkeypatch.setattr(factory, target, fail_initialization)
    app = create_app()

    with pytest.raises(ValueError, match="invalid embeddings configuration"):
        async with app.router.lifespan_context(app):
            pytest.fail("Application must not start when initialization fails")

    assert async_client.is_closed()
    assert sync_client.is_closed() == (provider_name == "langchain")
    if provider_name == "openai":
        sync_factory.assert_not_called()


def test_public_system_endpoints_are_available():
    client = TestClient(create_app())

    assert client.get("/").json() == {"message": "hello world"}
    assert client.get("/health").json() == {"status": "ok"}


def test_api_documentation_is_disabled_outside_debug_mode(monkeypatch):
    monkeypatch.setattr(config, "is_debug", False)
    client = TestClient(create_app())

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
