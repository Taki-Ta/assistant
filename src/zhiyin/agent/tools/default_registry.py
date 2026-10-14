from zhiyin.agent.tools.preferences import PreferenceTool
from zhiyin.agent.tools.registry import ToolRegistry
from zhiyin.agent.tools.search_knowledge import SearchKnowledgeTool

tool_registry = ToolRegistry(
    tools=[
        SearchKnowledgeTool(),
        PreferenceTool(),
        PreferenceTool(forget=True),
    ]
)
