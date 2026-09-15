from pydantic import BaseModel, ConfigDict
from uuid import UUID


class SourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: UUID
    content: str
    score: float
    document_id: UUID | None = None
    document_name: str | None = None
    headings: tuple[str, ...] = ()
    start_line: int | None = None
    end_line: int | None = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: tuple[SourceResponse, ...] = ()
