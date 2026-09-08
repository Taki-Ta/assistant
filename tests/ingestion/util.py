import hashlib
from datetime import UTC, datetime
from pathlib import Path

from interview_ai.ingestion.models import MarkdownFile


def make_file(content: str, path: Path | None = None) -> MarkdownFile:
    now = datetime.now(UTC)
    path = path or Path("knowledge.md")
    return MarkdownFile(
        path=path,
        name=path.name,
        create_time=now,
        modify_time=now,
        size=len(content.encode("utf-8")),
        content=content,
        hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
