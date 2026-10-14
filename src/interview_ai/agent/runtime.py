from dataclasses import dataclass
from uuid import UUID

from interview_ai.agent.context.tool_output import ToolOutputBudget
from interview_ai.agent.preferences import PreferenceService
from interview_ai.indexing.search_service import SearchService


@dataclass(frozen=True, slots=True)
class AgentContext:
    owner_id: str
    turn_id: UUID | None = None
    user_input: str = ""


@dataclass(frozen=True, slots=True)
class LangChainAgentContext:
    owner_id: str
    search_service: SearchService
    output_budget: ToolOutputBudget | None
    preference_service: PreferenceService | None = None
    turn_id: UUID | None = None
    user_input: str = ""


@dataclass(frozen=True, slots=True)
class ToolDependencies:
    search_service: SearchService
    preference_service: PreferenceService | None = None
