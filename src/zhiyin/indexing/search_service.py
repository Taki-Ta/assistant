from dataclasses import dataclass

from .models import SearchResult
from .protocols import EmbeddingProvider, VectorStore


@dataclass
class SearchService:
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    default_limit: int = 5
    score_threshold: float | None = 0.5

    def __post_init__(self) -> None:
        if self.default_limit <= 0:
            raise ValueError("default_limit 必须大于 0")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold 必须在 0 到 1 之间")

    async def search(
        self, query: str, owner_id: str, limit: int | None = None
    ) -> list[SearchResult]:
        result_limit = self.default_limit if limit is None else limit
        if result_limit <= 0:
            raise ValueError("limit 必须大于 0")
        query_vector = await self.embedding_provider.embed_one(query)
        return await self.vector_store.search(
            query_vector,
            owner_id=owner_id,
            limit=result_limit,
            score_threshold=self.score_threshold,
        )
