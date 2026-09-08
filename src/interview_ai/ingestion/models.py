from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5


@dataclass
class MarkdownFile:
    id: UUID = field(init=False)
    path: Path
    name: str
    create_time: datetime
    modify_time: datetime
    size: float
    content: str
    hash: str

    def __post_init__(self):
        self.id = uuid5(NAMESPACE_URL, str(self.path))


@dataclass
class Chunk:
    id: UUID = field(init=False)
    document_id: UUID
    sort_index: int
    level: int
    headings: tuple[str, ...]
    content: str
    start_line: int
    end_line: int
    hash: str

    def __post_init__(self):
        str = f"{self.document_id}-{self.sort_index}-{self.hash}"
        self.id = uuid5(NAMESPACE_URL, str)
