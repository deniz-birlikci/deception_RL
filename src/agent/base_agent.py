from abc import ABC, abstractmethod
from typing import Any
from ..models import (
    Backend,
    AIModel,
    MessageHistory,
    AssistantResponse,
)
from ..model_converters import BaseModelConverterFactory, ModelConverterFactoryRegistry
from ..env import settings
from ..models import Agent
from functools import wraps
from pathlib import Path
from datetime import datetime
import json


def log_messages(func):
    """Decorator to log messages sent to the model in a text file."""

    @wraps(func)
    async def wrapper(self, message_history: list[MessageHistory], *args, **kwargs):
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create a unique log file name based on the AI model
        log_file = logs_dir / f"{self.agent.agent_id}_messages.log"

        # Get current timestamp
        timestamp = datetime.now().isoformat()

        # Convert message history to a loggable format
        converted_history = self._convert_message_history(message_history)

        # Write to log file
        with open(log_file, "a") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(json.dumps(converted_history, indent=2))
            f.write("\n")
            f.write("=" * 80 + "\n\n")

        # Call the original function
        return await func(self, message_history, *args, **kwargs)

    return wrapper


class BaseAgent(ABC):
    backend: Backend

    def __init__(
        self,
        agent: Agent,
        ai_model: AIModel | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_converter_factory: BaseModelConverterFactory | None = None,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs

        self.agent = agent
        self.ai_model = ai_model or settings.ai.model_id
        self.api_key = api_key
        self.base_url = base_url

        new_model_factory = ModelConverterFactoryRegistry.create_factory(
            backend=self.backend,
            ai_model=self.ai_model,
        )
        self.model_converter_factory = model_converter_factory or new_model_factory

        self.user_input_converter = (
            self.model_converter_factory.create_user_input_converter()
        )
        self.tool_feedback_converter = (
            self.model_converter_factory.create_tool_feedback_converter()
        )
        self.assistant_response_converter = (
            self.model_converter_factory.create_assistant_response_converter()
        )

    @abstractmethod
    async def generate_response(
        self,
        message_history: list[MessageHistory],
        allowed_tools: list[str] | None = None,
        eligible_agent_ids: list[str] | None = None,
    ) -> AssistantResponse: ...

    def _convert_message_history(
        self, message_history: list[MessageHistory]
    ) -> list[dict]:
        return [
            item
            for message in message_history
            for item in self._convert_history_item(message)
        ]

    def _convert_history_item(self, message: MessageHistory) -> list[dict]:
        if message.history_type == "user-input":
            return self.user_input_converter.to_list_dict(message)
        elif message.history_type == "tool-feedback":
            return self.tool_feedback_converter.to_list_dict(message)
        elif message.history_type == "assistant-response":
            return self.assistant_response_converter.to_list_dict(message)
        else:
            raise ValueError(f"Unknown message history type: {message}")
