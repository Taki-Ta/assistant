from fastapi import APIRouter
from pydantic import BaseModel

from interview_ai.agent.providers.openai import OpenAIProvider

from ..dependencies import AgentContextDependency, ToolDependenciesDependency

router = APIRouter(prefix="/api/v1", tags=["ai"])
# DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
async def chat(
    input: ChatRequest,
    context: AgentContextDependency,
    dependencies: ToolDependenciesDependency,
) -> ChatResponse:
    async with OpenAIProvider() as provider:
        answer = await provider.generate(input.message, context, dependencies)
    return ChatResponse(answer=answer)
