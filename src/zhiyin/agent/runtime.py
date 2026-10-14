from dataclasses import dataclass
from uuid import UUID

from zhiyin.agent.preferences import PreferenceService
from zhiyin.indexing.search_service import SearchService


@dataclass(frozen=True, slots=True)
class AgentContext:
    owner_id: str
    turn_id: UUID | None = None
    user_input: str = ""


@dataclass(frozen=True, slots=True)
class ToolDependencies:
    search_service: SearchService
    preference_service: PreferenceService | None = None
