from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseFunctionToolCall
from uuid6 import uuid7

from interview_ai.agent.context.budget import (
    CompactionError,
    ContextBudget,
    ContextBudgetExceeded,
)
from interview_ai.agent.context.builder import ContextBuilder
from interview_ai.agent.context.compactor import (
    COMPACTION_INSTRUCTIONS,
    compact_history,
)
from interview_ai.agent.context.estimator import estimate_request, estimate_text
from interview_ai.agent.models import AgentMessage
from interview_ai.db.models import ConversationItem, ItemRole, ItemType


def test_text_estimate_handles_empty_ascii_chinese_and_mixed_text():
    assert estimate_text("") == 0
    assert estimate_text("abcdef") == 2
    assert estimate_text("中文") == 2
    assert estimate_text("abc中文") == 3


@pytest.mark.parametrize("total", [1024, 8192, 32768, 131072])
def test_default_reservations_scale_with_budget(total):
    budget = ContextBudget(total_tokens=total)
    assert budget.input_tokens > 0
    assert budget.initial_input_tokens > 0
    assert budget.output_tokens <= total // 4
    assert budget.safety_margin_tokens <= total // 8
    assert (
        budget.input_tokens + budget.output_tokens + budget.safety_margin_tokens
        == total
    )
    assert (
        budget.initial_input_tokens + budget.tool_output_tokens == budget.input_tokens
    )


@pytest.mark.parametrize("reserve", [-1, 800])
def test_invalid_tool_reservations_are_rejected(reserve):
    with pytest.raises(ValueError):
        ContextBudget(
            total_tokens=1000,
            output_tokens=100,
            safety_margin_tokens=100,
            tool_output_tokens=reserve,
        )


def test_explicit_invalid_reservations_are_rejected():
    with pytest.raises(ValueError, match="总预算不足"):
        ContextBudget(total_tokens=1000, output_tokens=1000, safety_margin_tokens=0)


@pytest.mark.asyncio
async def test_summary_network_error_has_safe_domain_message():
    summarizer = SimpleNamespace(
        summarize_history=AsyncMock(side_effect=RuntimeError("private-endpoint"))
    )
    with pytest.raises(CompactionError, match="历史摘要生成失败") as error:
        await compact_history(
            [AgentMessage(role="user", content="历史" * 100)],
            summarizer,
            ContextBudget(
                total_tokens=1500, output_tokens=200, safety_margin_tokens=100
            ),
            100,
        )
    assert "private-endpoint" not in str(error.value)


def test_request_counts_instructions_tools_and_growing_tool_history():
    messages = [AgentMessage(role="user", content="搜索知识库")]
    tools = [{"type": "function", "name": "search", "parameters": {}}]
    baseline = estimate_request("", messages)
    assert estimate_request("系统规则", messages) > baseline
    assert estimate_request("", messages, tools) > baseline
    call = ResponseFunctionToolCall(
        type="function_call",
        name="search",
        arguments='{"query":"中文"}',
        call_id="call-1",
    )
    result = {"type": "function_call_output", "call_id": "call-1", "output": "结果"}
    assert estimate_request("", [*messages, call, result], tools) > baseline
    assert estimate_request("", messages) == estimate_request(
        "", [{"role": "user", "content": "搜索知识库"}]
    )
    assert estimate_request("", [SimpleNamespace(type="message")]) > 0


@pytest.mark.asyncio
async def test_builder_preserves_history_and_appends_current_message_once():
    history = [
        ConversationItem(
            turn_id=uuid7(),
            sequence=index,
            item_type=ItemType.MESSAGE,
            role=role,
            text_content=text,
        )
        for index, (role, text) in enumerate(
            [
                (ItemRole.USER, "第一问"),
                (ItemRole.ASSISTANT, "第一答"),
                (ItemRole.USER, "未完成的问题"),
            ]
        )
    ]
    builder = ContextBuilder(instructions="规则")
    tools = [{"name": "search"}]
    built = await builder.build("继续", history, tools)
    assert [message.content for message in built.messages] == [
        "第一问",
        "第一答",
        "未完成的问题",
        "继续",
    ]
    assert built.instructions == "规则"
    assert built.estimated_input_tokens == estimate_request(
        "规则", built.messages, tools
    )
    assert len(history) == 3


@pytest.mark.asyncio
async def test_builder_rejects_history_without_text():
    item = ConversationItem(
        turn_id=uuid7(),
        sequence=0,
        item_type=ItemType.MESSAGE,
        role=ItemRole.USER,
    )
    with pytest.raises(ValueError, match="缺少 text_content"):
        await ContextBuilder().build("继续", [item])


def _history_turn(content: str) -> list[ConversationItem]:
    turn_id = uuid7()
    return [
        ConversationItem(
            turn_id=turn_id,
            sequence=index,
            item_type=ItemType.MESSAGE,
            role=role,
            text_content=text,
        )
        for index, (role, text) in enumerate(
            [
                (ItemRole.USER, content),
                (ItemRole.ASSISTANT, "回答"),
            ]
        )
    ]


@pytest.mark.asyncio
async def test_compaction_preserves_latest_whole_turn_and_records_coverage():
    older = _history_turn("旧内容" * 500)
    recent = _history_turn("最近的问题")
    summarizer = SimpleNamespace(
        summarize_history=AsyncMock(return_value="用户目标；未完成事项")
    )
    budget = ContextBudget(
        total_tokens=1500, output_tokens=200, safety_margin_tokens=100
    )
    builder = ContextBuilder(instructions="规则", summarizer=summarizer, budget=budget)
    built = await builder.build("继续", [*older, *recent])
    assert built.estimated_input_tokens <= budget.input_tokens
    assert built.messages[-3:] == [
        AgentMessage(role="user", content="最近的问题"),
        AgentMessage(role="assistant", content="回答"),
        AgentMessage(role="user", content="继续"),
    ]
    assert built.compaction.covered_through_turn_id == older[0].turn_id
    assert built.compaction.text == built.messages[0].content
    assert summarizer.summarize_history.await_count > 1
    for call in summarizer.summarize_history.await_args_list:
        cap = call.kwargs["max_output_tokens"]
        instructions = f"{COMPACTION_INSTRUCTIONS}\n摘要最多约 {cap} Token。"
        assert (
            estimate_request(instructions, call.args[0])
            <= budget.total_tokens - budget.safety_margin_tokens - cap
        )
    assert older[0].text_content == "旧内容" * 500


@pytest.mark.asyncio
async def test_fixed_input_overflow_does_not_call_summarizer():
    summarizer = SimpleNamespace(summarize_history=AsyncMock())
    builder = ContextBuilder(
        instructions="规则",
        summarizer=summarizer,
        budget=ContextBudget(
            total_tokens=1000, output_tokens=100, safety_margin_tokens=100
        ),
    )
    with pytest.raises(ContextBudgetExceeded, match="当前问题"):
        await builder.build("问题" * 1000, [])
    summarizer.summarize_history.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", ["", "过长" * 1000])
async def test_invalid_summary_is_rejected(summary):
    with pytest.raises(CompactionError):
        await compact_history(
            [AgentMessage(role="user", content="历史" * 300)],
            SimpleNamespace(summarize_history=AsyncMock(return_value=summary)),
            ContextBudget(
                total_tokens=1500, output_tokens=200, safety_margin_tokens=100
            ),
            100,
        )


@pytest.mark.asyncio
async def test_existing_checkpoint_is_merged_with_new_history():
    old_turn = uuid7()
    checkpoint = ConversationItem(
        turn_id=uuid7(),
        sequence=1,
        item_type=ItemType.COMPACTION,
        role=ItemRole.USER,
        text_content="已有历史摘要",
        payload={"covered_through_turn_id": str(old_turn)},
    )
    summarizer = SimpleNamespace(summarize_history=AsyncMock(return_value="合并摘要"))
    history = _history_turn("新历史" * 400)
    built = await ContextBuilder(
        instructions="规则",
        summarizer=summarizer,
        budget=ContextBudget(
            total_tokens=1500, output_tokens=200, safety_margin_tokens=100
        ),
    ).build("继续", [checkpoint, *history])
    assert built.compaction.covered_through_turn_id == history[0].turn_id
    assert (
        summarizer.summarize_history.await_args_list[0].args[0][0].content
        == "已有历史摘要"
    )


@pytest.mark.asyncio
async def test_builder_compacts_history_to_leave_tool_capacity():
    budget = ContextBudget(
        total_tokens=1200,
        output_tokens=100,
        safety_margin_tokens=100,
        tool_output_tokens=250,
    )
    history = [
        ConversationItem(
            turn_id=uuid7(),
            sequence=0,
            item_type=ItemType.MESSAGE,
            role=ItemRole.USER,
            text_content="历史" * 400,
        )
    ]
    messages = [
        AgentMessage(role="user", content=history[0].text_content),
        AgentMessage(role="user", content="继续"),
    ]
    original_size = estimate_request("规则", messages)
    assert budget.initial_input_tokens < original_size <= budget.input_tokens
    summarizer = SimpleNamespace(
        summarize_history=AsyncMock(return_value="较早问题摘要")
    )
    built = await ContextBuilder(
        instructions="规则",
        summarizer=summarizer,
        budget=budget,
    ).build("继续", history)
    assert built.compaction is not None
    assert built.estimated_input_tokens <= budget.initial_input_tokens
    assert built.messages[-1].content == "继续"
