from collections.abc import Iterable

from openai.types.responses.function_tool_param import FunctionToolParam

from ..protocols import AgentTool
from ..runtime import AgentContext, ToolDependencies
from ..models import ToolExecutionResult


class ToolRegistry:
    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self.tools: dict[str, AgentTool] = {}

        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"工具已注册：{tool.name}")
        else:
            self.tools[tool.name] = tool

    def definitions(self) -> list[FunctionToolParam]:
        return [tool.definition for tool in self.tools.values()]

    async def invoke(
        self,
        name: str,
        arguments: str,
        context: AgentContext,
        dependencies: ToolDependencies,
    ) -> ToolExecutionResult:
        tool = self.tools.get(name)

        if tool is None:
            raise LookupError(f"未知工具：{name}")
        return await tool.invoke(arguments, context, dependencies)
