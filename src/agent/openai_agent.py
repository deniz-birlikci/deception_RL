from typing import cast, Any
from .base_agent import BaseAgent, log_messages
from ..models import AIModel, MessageHistory, AssistantResponse, Backend, Agent
from ..model_converters import BaseModelConverterFactory
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from ..tools import generate_tools
from ..env import settings


class OpenAIAgent(BaseAgent):
    backend = Backend.OPENAI

    def __init__(
        self,
        agent: Agent,
        ai_model: AIModel | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_converter_factory: BaseModelConverterFactory | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent=agent,
            ai_model=ai_model,
            api_key=api_key,
            base_url=base_url,
            model_converter_factory=model_converter_factory,
            **kwargs,
        )

        self.api_key = self.api_key or settings.openai.api_key
        self.base_url = self.base_url or settings.openai.base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @log_messages
    async def generate_response(
        self,
        message_history: list[MessageHistory],
        allowed_tools: list[str] | None = None,
        eligible_agent_ids: list[str] | None = None,
    ) -> AssistantResponse:
        converted_history = self._convert_message_history(message_history)

        tools = generate_tools(allowed_tools, eligible_agent_ids)
        
        # Add reasoning field as first parameter to all tools
        from src.engine.protocol import add_reasoning_to_tool_schema
        tools = [add_reasoning_to_tool_schema(tool) for tool in tools]

        # DEBUG: Print message history to identify the issue
        # print("=== MESSAGE HISTORY BEING SENT TO OPENAI ===")
        # for i, msg in enumerate(converted_history):
        #     print(f"Message {i}: role={msg.get('role')}")
        #     if msg.get('role') == 'assistant' and 'tool_calls' in msg:
        #         print(f"  Has {len(msg['tool_calls'])} tool calls")
        #         for tc in msg['tool_calls']:
        #             print(f"    Tool call ID: {tc.get('id')}")
        #     elif msg.get('role') == 'tool':
        #         print(f"  Tool response for: {msg.get('tool_call_id')}")
        #     print(f"  Content: {str(msg.get('content', 'N/A'))[:100]}")
        # print("=== END MESSAGE HISTORY ===")

        response = await self.client.chat.completions.create(
            model=self.ai_model,
            messages=cast(list[ChatCompletionMessageParam], converted_history),
            tools=cast(list[ChatCompletionToolParam], tools),
            tool_choice="required",
        )

        assistant_response = self.assistant_response_converter.from_dict(
            data=response.model_dump()
        )

        assistant_response.tool_calls = assistant_response.tool_calls[:1]
        assistant_response.hydrated_tool_calls = assistant_response.hydrated_tool_calls[
            :1
        ]

        return assistant_response
