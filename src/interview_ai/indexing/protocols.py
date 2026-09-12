from typing import Protocol
from uuid import UUID

from interview_ai.db.models import Chunk, VectorRecord

from .models import SearchResult


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]: ...


class ChunkStore(Protocol):
    async def replace(self, document_id: UUID, chunks: list[Chunk]) -> None: ...


class VectorStore(Protocol):
    async def upsert(self, records: list[VectorRecord]) -> None: ...

    async def delete(self, chunk_ids: list[UUID]) -> None: ...

    async def search(
        self,
        query_vector: list[float],
        *,
        owner_id: str,
        limit: int = 5,
    ) -> list[SearchResult]: ...
