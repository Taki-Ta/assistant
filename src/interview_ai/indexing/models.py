from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass
class ChunkMetadata:
    path: Path
    headings: tuple[str, ...]
    start_line: int
    end_line: int


@dataclass
class VectorRecord:
    chunk_id: UUID
    vector: list[float]
    document_id: UUID
    content: str
    metadata: ChunkMetadata


@dataclass
class SearchResult:
    chunk_id: UUID
    content: str
    score: float
