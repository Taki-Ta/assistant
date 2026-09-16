from interview_ai.db.models import (
    ConversationItem,
    ItemRole,
    ItemType,
    Session,
    Turn,
    TurnStatus,
)


def test_agent_models_have_generated_ids_and_defaults() -> None:
    session = Session(owner_id="user-1")
    turn = Turn(session_id=session.id, sequence=0)
    item = ConversationItem(
        turn_id=turn.id,
        sequence=0,
        item_type=ItemType.MESSAGE,
        role=ItemRole.USER,
        text_content="什么是 RAG？",
    )

    assert session.id is not None
    assert session.is_deleted is False
    assert turn.id is not None
    assert turn.status == TurnStatus.IN_PROGRESS
    assert item.id is not None
    assert item.payload == {}


def test_conversation_item_supports_tool_calls() -> None:
    session = Session(owner_id="user-1")
    turn = Turn(session_id=session.id, sequence=0)
    item = ConversationItem(
        turn_id=turn.id,
        sequence=1,
        item_type=ItemType.FUNCTION_CALL,
        provider_item_id="fc_123",
        call_id="call_123",
        tool_name="search_knowledge",
        payload={"arguments": {"query": "Python 装饰器"}},
    )

    assert item.role is None
    assert item.call_id == "call_123"
    assert item.payload == {"arguments": {"query": "Python 装饰器"}}
