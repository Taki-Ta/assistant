from dataclasses import dataclass
from uuid import UUID


@dataclass
class SearchResult:
    chunk_id: UUID
    score: float
