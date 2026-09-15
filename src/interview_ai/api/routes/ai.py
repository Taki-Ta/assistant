from fastapi import APIRouter

from interview_ai.agent.providers.openai import OpenAIProvider

from ..dependencies import AgentContextDependency, ToolDependenciesDependency
from ..schemas import ChatRequest, ChatResponse, SourceResponse

router = APIRouter(prefix="/api/v1", tags=["ai"])


@router.post("/chat")
async def chat(
    input: ChatRequest,
    context: AgentContextDependency,
    dependencies: ToolDependenciesDependency,
) -> ChatResponse:
    async with OpenAIProvider() as provider:
        answer = await provider.generate(input.message, context, dependencies)
    return ChatResponse(
        answer=answer.answer,
        sources=tuple([SourceResponse.model_validate(item) for item in answer.sources]),
    )
