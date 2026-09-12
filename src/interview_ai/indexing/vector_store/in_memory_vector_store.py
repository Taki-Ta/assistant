from dataclasses import dataclass, field
from uuid import UUID

from interview_ai.db.models import VectorRecord
from interview_ai.util import cosine_similarity

from ..models import SearchResult


@dataclass
class InMemoryVectorStore:
    chunk_contents: dict[UUID, str] = field(default_factory=dict)
    chunk_owners: dict[UUID, str] = field(default_factory=dict)
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
        self,
        query_vector: list[float],
        *,
        owner_id: str,
        limit: int = 3,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        if not self.inner:
            return []
        results = [
            SearchResult(
                chunk_id=item.chunk_id,
                content=self.chunk_contents[item.chunk_id],
                score=cosine_similarity(query_vector, item.vector),
            )
            for item in self.inner
            if self.chunk_owners.get(item.chunk_id) == owner_id
            and item.chunk_id in self.chunk_contents
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
