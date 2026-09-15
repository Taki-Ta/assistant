from unittest.mock import AsyncMock

import pytest

from interview_ai.indexing.search_service import SearchService


@pytest.mark.asyncio
async def test_search_service_applies_configured_defaults() -> None:
    embedding_provider = AsyncMock()
    embedding_provider.embed_one.return_value = [0.1, 0.2]
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_limit=5,
        score_threshold=0.5,
    )

    await service.search("Python", "user-001")

    vector_store.search.assert_awaited_once_with(
        [0.1, 0.2],
        owner_id="user-001",
        limit=5,
        score_threshold=0.5,
    )


@pytest.mark.asyncio
async def test_search_service_allows_limit_override() -> None:
    embedding_provider = AsyncMock()
    embedding_provider.embed_one.return_value = [0.1, 0.2]
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_limit=5,
        score_threshold=0.5,
    )

    await service.search("Python", "user-001", limit=3)

    vector_store.search.assert_awaited_once_with(
        [0.1, 0.2],
        owner_id="user-001",
        limit=3,
        score_threshold=0.5,
    )
