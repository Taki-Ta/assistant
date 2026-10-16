from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from zhiyin.indexing.embeddings.langchain import LangChainEmbeddingProvider
from zhiyin.indexing.embeddings.openai import OpenAIEmbeddingProvider


@pytest.fixture(params=["openai", "langchain"])
def make_provider(request):
    def create(vectors):
        if request.param == "langchain":
            return LangChainEmbeddingProvider(
                SimpleNamespace(aembed_documents=AsyncMock(return_value=vectors))
            )
        response = SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(vectors)
            ]
        )
        return OpenAIEmbeddingProvider(
            client=SimpleNamespace(
                embeddings=SimpleNamespace(create=AsyncMock(return_value=response))
            ),
            model="test-model",
            dimensions=2,
        )

    return create


@pytest.mark.asyncio
async def test_embed_one_returns_valid_vector(make_provider):
    provider = make_provider([[0.1, 0.2]])

    assert await provider.embed_one("查询文本") == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [[], [[]], [[0.1, 0.2], [0.3, 0.4]]])
async def test_embed_one_rejects_invalid_response(make_provider, vectors):
    provider = make_provider(vectors)

    with pytest.raises(ValueError, match="单条 Embedding 请求必须返回一个非空向量"):
        await provider.embed_one("查询文本")
