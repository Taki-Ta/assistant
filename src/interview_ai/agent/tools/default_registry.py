from interview_ai.agent.tools.registry import ToolRegistry
from interview_ai.agent.tools.search_knowledge import SearchKnowledgeTool

tool_registry = ToolRegistry(
    tools=[
        SearchKnowledgeTool(),
    ]
)
