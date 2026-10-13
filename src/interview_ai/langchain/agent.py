import asyncio

from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, SecretStr
from interview_ai.agent.context.builder import INSTRUCTIONS
from interview_ai.config import config
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dataclass
class ModelContext:
    owner_id: str
    system_prompt: str


@dynamic_prompt
def build_system_prompt(request: ModelRequest) -> str:
    context = request.runtime.context
    return context.system_prompt


@tool
async def search_knowledge(
    query: str,
    runtime: ToolRuntime[ModelContext],
) -> str:
    """搜索当前用户的知识库。"""

    user_id = runtime.context.owner_id

    print(f"用户 {user_id} 正在检索：{query}")

    return "找到三条相关文档"


async def main():
    model = ChatOpenAI(
        model=config.chat_model,
        api_key=SecretStr(config.chat_api_key),
        base_url=config.chat_api_host,
        timeout=60,
        use_responses_api=True,
        output_version="responses/v1",
    )
    agent = create_agent(
        model=model,
        system_prompt=INSTRUCTIONS,
        context_schema=ModelContext,
        tools=[search_knowledge],
        middleware=[build_system_prompt],
    )
    async for mode, data in agent.astream(
        {"messages": [{"role": "user", "content": "查询知识库中的 Python 相关文档"}]},
        stream_mode=["messages", "updates"],
        context=ModelContext(owner_id="123", system_prompt=INSTRUCTIONS),
    ):
        if mode == "messages":
            chunk, metadata = data
            print("模型消息：", chunk)

        elif mode == "updates":
            print("状态更新：", data)


#     async for chunk in agent.astream(
#     {
#         "messages": [
#             {"role": "user", "content": "找找ai相关的内容"}
#         ]
#     },
#     stream_mode=["messages", "updates"],context=ModelContext(owner_id="123",system_prompt=INSTRUCTIONS),
# ):
#         print(chunk)
# result = await agent.stream(
#     {"messages": [HumanMessage("找找ai相关的内容")]},
#     context=ModelContext(owner_id="123",system_prompt=INSTRUCTIONS),
# )

# print(f"{result=}")


if __name__ == "__main__":
    asyncio.run(main())
