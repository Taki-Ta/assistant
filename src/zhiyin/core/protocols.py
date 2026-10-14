from typing import Protocol
from uuid import UUID


class IndexableDocument(Protocol):
    """切块和索引所需的最小文档接口。"""

    id: UUID
    name: str
    content: str
