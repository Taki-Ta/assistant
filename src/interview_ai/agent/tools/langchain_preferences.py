import json

from langchain.tools import ToolRuntime, tool
from pydantic import ConfigDict

from interview_ai.agent.preferences import (
    ForgetPreferenceArguments,
    RememberPreferenceArguments,
)

from ..runtime import AgentContext, LangChainAgentContext, ToolDependencies
from .preferences import PreferenceTool


class _RememberArguments(RememberPreferenceArguments):
    model_config = ConfigDict(extra="ignore")


class _ForgetArguments(ForgetPreferenceArguments):
    model_config = ConfigDict(extra="ignore")


@tool(args_schema=_RememberArguments)
async def remember_preference(
    key: str, value: str, evidence: str, runtime: ToolRuntime[LangChainAgentContext]
) -> str:
    """保存当前用户明确表达的长期偏好。key 为 response_language、answer_detail 或 code_language。

    response_language 支持 zh-CN/en/ja/ko/fr/de/es；answer_detail 支持 concise/balanced/detailed；
    code_language 使用 Python 等语言名称。evidence 必须是当前用户消息原文；本轮临时要求不保存。
    """
    result = await _invoke(
        runtime, False, {"key": key, "value": value, "evidence": evidence}
    )
    return result.output


@tool(args_schema=_ForgetArguments)
async def forget_preference(
    key: str, evidence: str, runtime: ToolRuntime[LangChainAgentContext]
) -> str:
    """删除当前用户明确要求忘记的偏好；evidence 必须引用当前用户消息原文。"""
    result = await _invoke(runtime, True, {"key": key, "evidence": evidence})
    return result.output


async def _invoke(runtime, forget, arguments):
    context = runtime.context
    return await PreferenceTool(forget).invoke(
        json.dumps(arguments, ensure_ascii=False),
        AgentContext(context.owner_id, context.turn_id, context.user_input),
        ToolDependencies(context.search_service, context.preference_service),
        output_budget=context.output_budget,
    )
