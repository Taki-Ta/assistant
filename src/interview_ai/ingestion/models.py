from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MarkdownFile:
    path: Path
    name: str
    create_time: datetime
    modify_time: datetime
    size: float
    content: str
