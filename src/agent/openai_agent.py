from typing import cast, Any
from .base_agent import BaseAgent
from ..models import AIModel, MessageHistory, AssistantResponse, Backend
from ..model_converters import BaseModelConverterFactory
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam
from ..tools import OPENAI_TOOLS
from ..env import settings


class OpenAIAgent(BaseAgent):
    backend = Backend.OPENAI

    def __init__(
        self,
        ai_model: AIModel | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_converter_factory: BaseModelConverterFactory | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ai_model,
            api_key,
            base_url,
            model_converter_factory,
            **kwargs,
        )

        self.api_key = self.api_key or settings.openai.api_key
        self.base_url = self.base_url or settings.openai.base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def generate_response(
        self,
        message_history: list[MessageHistory],
        allowed_tools: list[str] | None = None,
    ) -> AssistantResponse:
        converted_history = self._convert_message_history(message_history)

        tools = (
            [t for t in OPENAI_TOOLS if t["function"]["name"] in allowed_tools]
            if allowed_tools
            else OPENAI_TOOLS
        )

        response = await self.client.chat.completions.create(
            model=self.ai_model,
            messages=cast(list[ChatCompletionMessageParam], converted_history),
            tools=cast(list[ChatCompletionToolUnionParam], tools),
            tool_choice="required",
        )

        return self.assistant_response_converter.from_dict(data=response.model_dump())
