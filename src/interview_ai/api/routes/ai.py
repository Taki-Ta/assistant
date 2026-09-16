from fastapi import APIRouter, HTTPException, status

from interview_ai.agent.services import SessionNotFoundError

from ..dependencies import (
    AgentContextDependency,
    ConversationServiceDependency,
    ToolDependenciesDependency,
)
from ..schemas import ChatRequest, ChatResponse, SourceResponse

router = APIRouter(prefix="/api/v1", tags=["ai"])


@router.post("/chat")
async def chat(
    input: ChatRequest,
    service: ConversationServiceDependency,
    context: AgentContextDependency,
    dependencies: ToolDependenciesDependency,
) -> ChatResponse:
    try:
        answer = await service.chat(
            input.session_id,
            input.message,
            context,
            dependencies,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    return ChatResponse(
        answer=answer.answer,
        retrieved_sources=tuple(
            SourceResponse.model_validate(item) for item in answer.retrieved_sources
        ),
        session_id=answer.session_id,
    )
