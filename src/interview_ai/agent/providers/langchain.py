

from collections.abc import AsyncGenerator

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest,dynamic_prompt
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy import Sequence
from interview_ai.agent.context.builder import INSTRUCTIONS
from interview_ai.config import config

from interview_ai.agent.models import AgentEvent, AgentMessage
from interview_ai.agent.runtime import AgentContext, ToolDependencies


class LangChainProvider:

    def __init__(self) -> None:
        model = ChatOpenAI(
               model=config.chat_model,
               api_key=SecretStr(config.chat_api_key),
               base_url=config.chat_api_host,
               timeout=60,
               use_responses_api=True,
               output_version="responses/v1",
           )
        self.agent = create_agent(model=model)

    @dynamic_prompt
    def build_system_prompt(request:ModelRequest):
         context=request.runtime.context
     
    def generate(
            self,
            messages: Sequence[AgentMessage],
            context: AgentContext,
            dependencies: ToolDependencies,
            instructions: str | None = None,
            *,
            stream: bool = False,
        ) -> AsyncGenerator[AgentEvent, None]:
          yield 