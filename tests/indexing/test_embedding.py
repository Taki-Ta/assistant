import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from zhiyin.config import config
from zhiyin.indexing.embeddings.factory import create_embedding_provider
from zhiyin.indexing.embeddings.openai import OpenAIEmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_provider_splits_large_inputs_into_ordered_batches():
    create = AsyncMock(
        side_effect=[
            SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[2.0]),
                    SimpleNamespace(index=0, embedding=[1.0]),
                ]
            ),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[3.0])]),
        ]
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    provider = OpenAIEmbeddingProvider(
        client=client,
        model="embedding-model",
        dimensions=1,
        batch_size=2,
    )

    vectors = await provider.embed(["one", "two", "three"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert [call.kwargs["input"] for call in create.await_args_list] == [
        ["one", "two"],
        ["three"],
    ]


@pytest_asyncio.fixture(params=["openai", "langchain"])
async def provider(request):
    settings = config.model_copy(
        update={"embedding_provider": request.param, "embedding_batch_size": 2}
    )
    async with create_embedding_provider(settings) as embedding_provider:
        yield embedding_provider


@pytest.mark.network
@pytest.mark.asyncio
async def test_embedding_provider_should_work(provider):
    texts = ["风急天高猿啸哀", "渚清沙白鸟飞回", "无边落木萧萧下", "不尽长江滚滚来"]

    async with asyncio.timeout(60):
        result = await provider.embed(texts)
        query_vector = await provider.embed_one("诗歌中的秋景")

    assert provider.model_name == config.embedding_model_name
    assert len(result) == len(texts)
    assert all(len(vector) == config.dimensions for vector in result)
    assert all(math.isfinite(value) for vector in result for value in vector)
    assert len(query_vector) == config.dimensions
    assert all(math.isfinite(value) for value in query_vector)
