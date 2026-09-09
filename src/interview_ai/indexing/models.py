from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from .searching import cosine_similarity


@dataclass
class ChunkMetadata:
    path: Path
    headings: list[str]
    start_line: int
    end_line: int


@dataclass
class VectorRecord:
    chunk_id: UUID
    vector: list[float]
    # document_id: UUID
    content: str
    # metadata: ChunkMetadata


@dataclass
class SearchResult:
    chunk_id: UUID
    content: str
    score: float


@dataclass
class InMemoryVectorStore:
    inner: list[VectorRecord] = field(init=False, default_factory=list)

    def upsert(self, records: list[VectorRecord]) -> None:
        """更新或新增向量"""
        for record in records:
            for index, stored_record in enumerate(self.inner):
                if stored_record.chunk_id == record.chunk_id:
                    self.inner[index] = record
                    break
            else:
                self.inner.append(record)

    def delete(self, chunk_ids: list[UUID]) -> None:
        ids_to_delete = set(chunk_ids)
        self.inner = [
            record for record in self.inner if record.chunk_id not in ids_to_delete
        ]

    def search(self, query_vector: list[float], limit: int = 3) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        if not self.inner:
            return []
        results = [
            SearchResult(
                item.chunk_id,
                item.content,
                cosine_similarity(query_vector, item.vector),
            )
            for item in self.inner
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
