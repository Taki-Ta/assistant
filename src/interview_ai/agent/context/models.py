from dataclasses import dataclass
from uuid import UUID

from interview_ai.agent.models import AgentMessage


@dataclass(frozen=True)
class CompactionCheckpoint:
    text: str
    covered_through_turn_id: UUID


@dataclass(frozen=True)
class BuiltContext:
    instructions: str
    messages: list[AgentMessage]
    estimated_input_tokens: int
    compaction: CompactionCheckpoint | None = None
