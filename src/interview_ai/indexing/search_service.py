from dataclasses import dataclass

from .models import SearchResult
from .protocols import EmbeddingProvider, VectorStore


@dataclass
class SearchService:
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore

    async def search(
        self, query: str, owner_id: str, limit: int = 5
    ) -> list[SearchResult]:
        query_vector = await self.embedding_provider.embed_one(query)
        return await self.vector_store.search(
            query_vector,
            owner_id=owner_id,
            limit=limit,
        )
