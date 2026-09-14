from dataclasses import dataclass

from interview_ai.indexing.search_service import SearchService


@dataclass(frozen=True, slots=True)
class AgentContext:
    owner_id: str


@dataclass(frozen=True, slots=True)
class ToolDependencies:
    search_service: SearchService
