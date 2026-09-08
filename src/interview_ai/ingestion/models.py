from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4


@dataclass
class MarkdownFile:
    id: UUID = field(init=False, default_factory=uuid4)
    path: Path
    name: str
    create_time: datetime
    modify_time: datetime
    size: float
    content: str
    hash: str


@dataclass
class Chunk:
    id: UUID = field(init=False, default_factory=uuid4)
    file_id: UUID
    sortindex: int
    level: int
    headings: tuple[str, ...]
    content: str
    start_index: int
    end_index: int
    hash: str
