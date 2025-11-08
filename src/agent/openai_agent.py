from typing import cast, Any
from .base_agent import BaseAgent
from ..models import AIModel, MessageHistory, AssistantResponse, Backend
from ..model_converters import BaseModelConverterFactory
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam
from ..tools import generate_tools
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
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.provider_order = self.kwargs.get(
            "provider_order", settings.openai.provider_order
        )
        self.reasoning_enabled = self.kwargs.get(
            "reasoning_enabled", settings.openai.reasoning_enabled
        )
        self.allow_fallbacks = self.kwargs.get(
            "allow_fallbacks", settings.openai.allow_fallbacks
        )

    def generate_response(
        self,
        message_history: list[MessageHistory],
        allowed_tools: list[str] | None = None,
        eligible_agent_ids: list[str] | None = None,
    ) -> AssistantResponse:
        converted_history = self._convert_message_history(message_history)

        tools = generate_tools(allowed_tools, eligible_agent_ids)

        reasoning_enabled_dict = {
            "enabled": self.reasoning_enabled,
        }

        provider_dict = {
            "order": self.provider_order,
            "allow_fallbacks": self.allow_fallbacks,
        }

        response = self.client.chat.completions.create(
            model=self.ai_model,
            messages=cast(list[ChatCompletionMessageParam], converted_history),
            tools=cast(list[ChatCompletionToolUnionParam], tools),
            tool_choice="required",
            extra_body={
                "provider": provider_dict,
                "reasoning": reasoning_enabled_dict,
            },
        )

        print(response.model_dump())

        return self.assistant_response_converter.from_dict(data=response.model_dump())
