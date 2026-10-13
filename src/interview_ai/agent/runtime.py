from dataclasses import dataclass

from interview_ai.agent.context.tool_output import ToolOutputBudget
from interview_ai.indexing.search_service import SearchService


@dataclass(frozen=True, slots=True)
class AgentContext:
    owner_id: str


@dataclass(frozen=True, slots=True)
class LangChainAgentContext:
    owner_id: str
    search_service: SearchService
    output_budget: ToolOutputBudget | None


@dataclass(frozen=True, slots=True)
class ToolDependencies:
    search_service: SearchService
