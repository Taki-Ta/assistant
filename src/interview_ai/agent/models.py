from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from uuid import UUID


# @dataclass(frozen=True, slots=True)
class RetrievedSource(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: UUID
    content: str
    score: float
    document_id: UUID | None = None
    document_name: str | None = None
    headings: tuple[str, ...] = ()
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sources: tuple[RetrievedSource, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    output: str
    sources: tuple[RetrievedSource, ...] = ()
