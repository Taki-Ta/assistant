import math

import pytest
from openai import OpenAI

from interview_ai.config import config
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider


@pytest.fixture
def provider():
    with OpenAI(
        api_key=config.api_key,
        base_url=config.host,
    ) as client:
        yield OpenAIEmbeddingProvider(
            client=client,
            model=config.embedding_model_name,
            dimensions=config.dimensions,
        )


def test_openai_embedding_provider_should_work(provider):
    texts = ["风急天高猿啸哀", "渚清沙白鸟飞回", "无边落木萧萧下", "不尽长江滚滚来"]

    result = provider.embed(texts)

    assert len(result) == len(texts)
    assert all(len(vector) == config.dimensions for vector in result)
    assert all(math.isfinite(value) for vector in result for value in vector)
