from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_openai import OpenAIEmbeddings

from zhiyin.indexing.embeddings.langchain import LangChainEmbeddingProvider


@pytest.fixture
def embedding_client():
    # Keep the real LangChain interface and batching, replacing only API calls.
    return OpenAIEmbeddings(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=2,
        chunk_size=2,
        check_embedding_ctx_length=False,
        client=Mock(),
        async_client=SimpleNamespace(create=AsyncMock()),
    )


def test_model_name(embedding_client):
    provider = LangChainEmbeddingProvider(embedding_client)

    assert provider.model_name == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_empty_input_does_not_call_api(embedding_client):
    provider = LangChainEmbeddingProvider(embedding_client)

    assert await provider.embed([]) == []
    embedding_client.async_client.create.assert_not_called()
    embedding_client.client.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 3, 5])
async def test_embed_batches_inputs_and_preserves_order(embedding_client, count):
    texts = [f"text-{index}" for index in range(count)]
    vectors = [[float(index), 1.0] for index in range(count)]
    create = embedding_client.async_client.create
    create.side_effect = [
        {"data": [{"embedding": vector} for vector in vectors[start : start + 2]]}
        for start in range(0, count, 2)
    ]
    provider = LangChainEmbeddingProvider(embedding_client)

    assert await provider.embed(texts) == vectors

    assert [call.kwargs["input"] for call in create.await_args_list] == [
        texts[start : start + 2] for start in range(0, count, 2)
    ]
    assert all(
        call.kwargs["model"] == embedding_client.model
        and call.kwargs["dimensions"] == 2
        for call in create.await_args_list
    )
    embedding_client.client.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["查询文本", ""])
async def test_embed_one_returns_single_vector(embedding_client, text):
    create = embedding_client.async_client.create
    create.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    provider = LangChainEmbeddingProvider(embedding_client)

    assert await provider.embed_one(text) == [0.1, 0.2]

    create.assert_awaited_once()
    assert create.await_args.kwargs["input"] == [text]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, value", [("embed", ["text"]), ("embed_one", "text")])
async def test_api_error_propagates(embedding_client, method, value):
    embedding_client.async_client.create.side_effect = RuntimeError("API unavailable")
    provider = LangChainEmbeddingProvider(embedding_client)

    with pytest.raises(RuntimeError, match="API unavailable"):
        await getattr(provider, method)(value)
