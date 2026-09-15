from dataclasses import dataclass

from openai.types.responses.function_tool_param import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from interview_ai.indexing.models import SearchResult

from ..models import RetrievedSource, ToolExecutionResult
from ..runtime import AgentContext, ToolDependencies


class SearchKnowledgeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="需要在用户知识库中检索的问题或关键词",
    )


@dataclass
class SearchKnowledgeTool:
    @property
    def name(self) -> str:
        return "search_knowledge"

    @property
    def definition(self) -> FunctionToolParam:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                "搜索当前用户的知识库。当回答面试题、技术概念或用户资料相关问题时使用。"
            ),
            "strict": True,
            "parameters": SearchKnowledgeArguments.model_json_schema(),
        }

    async def invoke(
        self, arguments: str, context: AgentContext, dependencies: ToolDependencies
    ) -> ToolExecutionResult:
        args = SearchKnowledgeArguments.model_validate_json(arguments)

        results = await dependencies.search_service.search(
            query=args.query,
            owner_id=context.owner_id,
        )

        return ToolExecutionResult(
            TypeAdapter(list[SearchResult]).dump_json(results).decode("utf-8"),
            tuple(RetrievedSource.model_validate(item) for item in results),
        )
