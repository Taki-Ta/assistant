from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RelevantSource(BaseModel):
    document: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    contains: str = Field(min_length=1)


class RetrievalCase(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_sources: list[RelevantSource] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class EvaluatedMatch(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID | None
    document_name: str | None
    headings: tuple[str, ...]
    content: str
    start_line: int | None
    end_line: int | None
    score: float
    relevant: bool


class RetrievalCaseResult(BaseModel):
    id: str
    query: str
    tags: list[str]
    latency_ms: float
    first_relevant_rank: int | None
    recall_at_k: float
    results: list[EvaluatedMatch]


class RetrievalMetrics(BaseModel):
    hit_at_1: float
    hit_at_k: float
    mrr_at_k: float
    recall_at_k: float
    average_latency_ms: float
    p95_latency_ms: float


class RetrievalReport(BaseModel):
    created_at: datetime
    embedding_model: str
    dimensions: int = Field(gt=0)
    limit: int = Field(gt=0)
    question_count: int = Field(ge=0)
    metrics: RetrievalMetrics
    cases: list[RetrievalCaseResult]
