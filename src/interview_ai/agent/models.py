from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


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
    session_id: UUID
    answer: str
    retrieved_sources: tuple[RetrievedSource, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    output: str
    sources: tuple[RetrievedSource, ...] = ()


class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_response_id: str | None = None
    provider_item_id: str | None = None


class AssistantMessageEvent(BaseEvent):
    type: Literal["message"] = "message"
    text: str


class FunctionCallEvent(BaseEvent):
    type: Literal["function_call"] = "function_call"
    call_id: str
    tool_name: str
    arguments: dict[str, object]


class FunctionCallOutputEvent(BaseEvent):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    tool_name: str
    output: str
    succeeded: bool
    sources: tuple[RetrievedSource, ...] = ()


class ModelCompletedEvent(BaseEvent):
    type: Literal["model_completed"] = "model_completed"
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


ConversationEvent = AssistantMessageEvent | FunctionCallEvent | FunctionCallOutputEvent
AgentEvent = ConversationEvent | ModelCompletedEvent
