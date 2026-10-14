import json
from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool
from openai.types.responses.function_tool_param import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from zhiyin.indexing.models import SearchResult

from ..context.budget import ContextBudgetExceeded
from ..context.tool_output import ToolOutputBudget
from ..models import RetrievedSource, ToolExecutionResult
from ..runtime import AgentContext, LangChainAgentContext, ToolDependencies


class SearchKnowledgeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="需要在用户知识库中检索的问题或关键词",
    )


class _LangChainSearchArguments(SearchKnowledgeArguments):
    # LangChain validates injected runtime alongside model arguments, then
    # restores it from the original input. It must not be rejected as extra.
    model_config = ConfigDict(extra="ignore")


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
        self,
        arguments: str,
        context: AgentContext,
        dependencies: ToolDependencies,
        *,
        output_budget: ToolOutputBudget | None = None,
    ) -> ToolExecutionResult:
        args = SearchKnowledgeArguments.model_validate_json(arguments)

        search_context = LangChainAgentContext(
            owner_id=context.owner_id,
            search_service=dependencies.search_service,
            output_budget=output_budget,
        )

        return await _search_knowledgebase(
            query=args.query,
            owner_id=context.owner_id,
            context=search_context,
        )


async def _search_knowledgebase(
    query: str,
    owner_id: str,
    context: LangChainAgentContext,
) -> ToolExecutionResult:
    results = await context.search_service.search(
        query=query,
        owner_id=owner_id,
    )
    output_budget = context.output_budget
    output = TypeAdapter(list[SearchResult]).dump_json(results).decode("utf-8")
    sources = [RetrievedSource.model_validate(item) for item in results]
    if output_budget is None or output_budget.fits(output):
        return ToolExecutionResult(output, tuple(sources))

    def serialize(selected: list[RetrievedSource]) -> str:
        return json.dumps(
            {
                "results": [source.model_dump(mode="json") for source in selected],
                "truncated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    selected: list[RetrievedSource] = []
    ranked = sorted(sources, key=lambda source: source.score, reverse=True)
    for source in ranked:
        if output_budget.fits(serialize([*selected, source])):
            selected.append(source)
    if not selected and ranked:
        source = ranked[0]
        low, high = 0, len(source.content)
        while low < high:
            mid = (low + high + 1) // 2
            fragment = source.model_copy(
                update={
                    "content": source.content[:mid] + "\n[片段内容已截断]",
                    "end_line": None,
                }
            )
            if output_budget.fits(serialize([fragment])):
                low = mid
            else:
                high = mid - 1
        if low > 0:
            selected.append(
                source.model_copy(
                    update={
                        "content": source.content[:low] + "\n[片段内容已截断]",
                        "end_line": None,
                    }
                )
            )
    output = serialize(selected)
    if (ranked and not selected) or not output_budget.fits(output):
        raise ContextBudgetExceeded("剩余预算不足以容纳有效检索片段")
    return ToolExecutionResult(output, tuple(selected))


@tool(
    "search_knowledge",
    args_schema=_LangChainSearchArguments,
    response_format="content_and_artifact",
)
async def search_knowledgebase(
    query: str,
    runtime: ToolRuntime[LangChainAgentContext],
) -> tuple[str, dict[str, object]]:
    """检索当前用户的知识库。"""
    result = await _search_knowledgebase(
        query=query,
        owner_id=runtime.context.owner_id,
        context=runtime.context,
    )
    return result.output, {
        "sources": [source.model_dump(mode="json") for source in result.sources]
    }
