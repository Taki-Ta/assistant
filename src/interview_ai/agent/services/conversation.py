from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.agent.context.builder import ContextBuilder
from interview_ai.agent.models import (
    AgentMessage,
    AgentResult,
    AssistantMessageEvent,
    ConversationEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
    RetrievedSource,
)
from interview_ai.agent.protocols import AgentProvider
from interview_ai.agent.repository import ConversationRepository
from interview_ai.agent.runtime import AgentContext, ToolDependencies
from interview_ai.db.models import ConversationItem, ItemRole, ItemType, utc_now


class SessionNotFoundError(LookupError):
    pass


@dataclass
class ConversationService:
    _db: AsyncSession
    _repository: ConversationRepository
    _provider: AgentProvider
    _context_builder: ContextBuilder

    async def chat(
        self,
        session_id: UUID | None,
        message: str,
        context: AgentContext,
        dependencies: ToolDependencies,
    ) -> AgentResult:
        current_message = AgentMessage(role="user", content=message)

        async with self._db.begin():
            if session_id is None:
                session = await self._repository.create_session(context.owner_id)
            else:
                session = await self._repository.get_session(
                    session_id,
                    context.owner_id,
                    for_update=True,
                )
                if session is None:
                    raise SessionNotFoundError(str(session_id))

            turn = await self._repository.create_turn(session.id)
            history_items = await self._repository.list_context_history(
                session.id,
                before_turn_sequence=turn.sequence,
            )
            await self._repository.append_items(
                [
                    ConversationItem(
                        turn_id=turn.id,
                        sequence=0,
                        item_type=ItemType.MESSAGE,
                        role=ItemRole.USER,
                        text_content=current_message.content,
                    )
                ]
            )

        conversation_events: list[ConversationEvent] = []
        retrieved_sources: dict[UUID, RetrievedSource] = {}
        answer: str | None = None
        completed: ModelCompletedEvent | None = None
        first_event_sequence = 1

        try:
            built_context = await self._context_builder.build(
                message, history_items, self._provider.tool_definitions()
            )
            if built_context.compaction is not None:
                checkpoint = built_context.compaction
                # 摘要调用已在事务外完成；先持久化，后续回答失败仍能恢复摘要。
                async with self._db.begin():
                    await self._repository.append_items(
                        [
                            ConversationItem(
                                turn_id=turn.id,
                                sequence=1,
                                item_type=ItemType.COMPACTION,
                                role=ItemRole.USER,
                                text_content=checkpoint.text,
                                payload={
                                    "covered_through_turn_id": str(
                                        checkpoint.covered_through_turn_id
                                    ),
                                    "estimated_input_tokens": built_context.estimated_input_tokens,
                                },
                            )
                        ]
                    )
                first_event_sequence = 2
            async for event in self._provider.generate(
                built_context.messages,
                context,
                dependencies,
                instructions=built_context.instructions,
            ):
                if isinstance(event, ModelCompletedEvent):
                    completed = event
                    continue

                conversation_events.append(event)

                if isinstance(event, AssistantMessageEvent):
                    answer = event.text
                elif isinstance(event, FunctionCallOutputEvent) and event.succeeded:
                    _merge_sources(retrieved_sources, event.sources)

            if answer is None or completed is None:
                raise RuntimeError("模型响应未正常完成")

        except Exception as exc:
            async with self._db.begin():
                await self._append_events(
                    turn.id, conversation_events, first_event_sequence
                )
                await self._repository.fail_turn(
                    turn,
                    error_message=str(exc),
                    completed_at=utc_now(),
                )
            raise

        async with self._db.begin():
            await self._append_events(
                turn.id, conversation_events, first_event_sequence
            )
            await self._repository.complete_turn(
                turn,
                provider=completed.provider,
                model=completed.model,
                response_id=completed.provider_response_id,
                input_tokens=completed.input_tokens,
                output_tokens=completed.output_tokens,
                total_tokens=completed.total_tokens,
                completed_at=utc_now(),
            )

        sources = tuple(
            sorted(
                retrieved_sources.values(),
                key=lambda source: source.score,
                reverse=True,
            )
        )
        return AgentResult(
            session_id=session.id,
            answer=answer,
            retrieved_sources=sources,
        )

    async def _append_events(
        self,
        turn_id: UUID,
        events: list[ConversationEvent],
        start_sequence: int = 1,
    ) -> None:
        if not events:
            return

        await self._repository.append_items(
            [
                _event_to_item(turn_id, sequence, event)
                for sequence, event in enumerate(events, start=start_sequence)
            ]
        )


def _merge_sources(
    target: dict[UUID, RetrievedSource],
    sources: tuple[RetrievedSource, ...],
) -> None:
    for source in sources:
        existing = target.get(source.chunk_id)
        if existing is None or source.score > existing.score:
            target[source.chunk_id] = source


def _event_to_item(
    turn_id: UUID,
    sequence: int,
    event: ConversationEvent,
) -> ConversationItem:
    if isinstance(event, AssistantMessageEvent):
        return ConversationItem(
            turn_id=turn_id,
            sequence=sequence,
            item_type=ItemType.MESSAGE,
            role=ItemRole.ASSISTANT,
            provider_item_id=event.provider_item_id,
            text_content=event.text,
            payload={
                "provider_response_id": event.provider_response_id,
            },
        )

    if isinstance(event, FunctionCallEvent):
        return ConversationItem(
            turn_id=turn_id,
            sequence=sequence,
            item_type=ItemType.FUNCTION_CALL,
            provider_item_id=event.provider_item_id,
            call_id=event.call_id,
            tool_name=event.tool_name,
            payload={
                "provider_response_id": event.provider_response_id,
                "arguments": event.arguments,
            },
        )

    return ConversationItem(
        turn_id=turn_id,
        sequence=sequence,
        item_type=ItemType.FUNCTION_CALL_OUTPUT,
        provider_item_id=event.provider_item_id,
        call_id=event.call_id,
        tool_name=event.tool_name,
        text_content=event.output,
        payload={
            "provider_response_id": event.provider_response_id,
            "succeeded": event.succeeded,
            "sources": [
                {
                    "chunk_id": str(source.chunk_id),
                    "document_id": (
                        str(source.document_id)
                        if source.document_id is not None
                        else None
                    ),
                    "score": source.score,
                }
                for source in event.sources
            ],
        },
    )
