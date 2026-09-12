from dataclasses import dataclass, field
from uuid import UUID

from interview_ai.db.models import VectorRecord

from ..models import SearchResult
from ..searching import cosine_similarity


@dataclass
class InMemoryVectorStore:
    inner: list[VectorRecord] = field(init=False, default_factory=list)

    async def upsert(self, records: list[VectorRecord]) -> None:
        """更新或新增向量。"""
        for record in records:
            for index, stored_record in enumerate(self.inner):
                if stored_record.chunk_id == record.chunk_id:
                    self.inner[index] = record
                    break
            else:
                self.inner.append(record)

    async def delete(self, chunk_ids: list[UUID]) -> None:
        ids_to_delete = set(chunk_ids)
        self.inner = [
            record for record in self.inner if record.chunk_id not in ids_to_delete
        ]

    async def search(
        self, query_vector: list[float], limit: int = 3
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        if not self.inner:
            return []
        results = [
            SearchResult(
                item.chunk_id,
                cosine_similarity(query_vector, item.vector),
            )
            for item in self.inner
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
