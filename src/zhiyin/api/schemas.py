from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from zhiyin.db.models import TurnStatus


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
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    answer: str
    retrieved_sources: tuple[SourceResponse, ...] = ()
    session_id: UUID


class ChatErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    session_id: UUID
    turn_id: UUID


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    items: list[SessionResponse]
    next_cursor: str | None


class MessageItemResponse(BaseModel):
    id: UUID
    sequence: int
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    text: str


class FunctionCallItemResponse(BaseModel):
    id: UUID
    sequence: int
    type: Literal["function_call"] = "function_call"
    call_id: str
    tool_name: str
    arguments: dict[str, object]


class FunctionCallOutputItemResponse(BaseModel):
    id: UUID
    sequence: int
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    tool_name: str
    output: str
    succeeded: bool
    sources: list[SourceResponse]


class TurnResponse(BaseModel):
    id: UUID
    sequence: int
    status: TurnStatus
    created_at: datetime
    completed_at: datetime | None
    error_message: str | None
    items: list[
        Annotated[
            MessageItemResponse
            | FunctionCallItemResponse
            | FunctionCallOutputItemResponse,
            Field(discriminator="type"),
        ]
    ]


class TurnListResponse(BaseModel):
    items: list[TurnResponse]
    next_cursor: int | None
