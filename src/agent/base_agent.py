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

    # ANSI color codes
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def wrap_text(text: str, width: int = 120) -> str:
        """Wrap text to specified width, preserving paragraphs."""
        if not text:
            return ""

        # Handle different types of content
        if isinstance(text, dict) or isinstance(text, list):
            text = json.dumps(text, indent=2)
        else:
            text = str(text)

        # Simple line wrapping at specified width
        lines = []
        for line in text.split("\n"):
            if len(line) <= width:
                lines.append(line)
            else:
                # Wrap long lines
                while len(line) > width:
                    lines.append(line[:width])
                    line = line[width:]
                if line:
                    lines.append(line)
        return "\n".join(lines)

    def format_message_pretty(message: dict, role_color: str, role_name: str) -> str:
        """Format a single message with color and text wrapping."""
        output = []
        output.append(f"{BOLD}{role_color}{'=' * 120}{RESET}")
        output.append(f"{BOLD}{role_color}[{role_name.upper()}]{RESET}")
        output.append(f"{role_color}{'-' * 120}{RESET}")

        # Handle different message structures
        if not isinstance(message, dict):
            wrapped = wrap_text(message)
            output.append(f"{role_color}{wrapped}{RESET}")
        elif "content" in message:
            content = message["content"]
            if isinstance(content, str):
                wrapped = wrap_text(content)
                output.append(f"{role_color}{wrapped}{RESET}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            wrapped = wrap_text(item.get("text", ""))
                            output.append(f"{role_color}{wrapped}{RESET}")
                        elif item.get("type") == "tool_use":
                            output.append(
                                f"{role_color}[TOOL USE: {item.get('name', 'unknown')}]{RESET}"
                            )
                            wrapped = wrap_text(item.get("input", {}))
                            output.append(f"{role_color}{wrapped}{RESET}")
                        elif item.get("type") == "tool_result":
                            output.append(
                                f"{role_color}[TOOL RESULT: {item.get('tool_use_id', 'unknown')}]{RESET}"
                            )
                            wrapped = wrap_text(item.get("content", ""))
                            output.append(f"{role_color}{wrapped}{RESET}")
                        else:
                            wrapped = wrap_text(item)
                            output.append(f"{role_color}{wrapped}{RESET}")
                    else:
                        wrapped = wrap_text(item)
                        output.append(f"{role_color}{wrapped}{RESET}")
            else:
                wrapped = wrap_text(content)
                output.append(f"{role_color}{wrapped}{RESET}")
        else:
            # Fallback for other structures
            wrapped = wrap_text(message)
            output.append(f"{role_color}{wrapped}{RESET}")

        output.append(f"{role_color}{'=' * 120}{RESET}")
        output.append("")
        return "\n".join(output)

    @wraps(func)
    async def wrapper(self, message_history: list[MessageHistory], *args, **kwargs):
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create log file names based on the AI model
        log_file = logs_dir / f"{self.agent.agent_id}_messages.log"
        pretty_log_file = logs_dir / f"{self.agent.agent_id}_messages_pretty.log"

        # Get current timestamp
        timestamp = datetime.now().isoformat()

        # Convert message history to a loggable format
        converted_history = self._convert_message_history(message_history)

        # Write to JSON log file
        with open(log_file, "a") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(json.dumps(converted_history, indent=2))
            f.write("\n")
            f.write("=" * 80 + "\n\n")

        # Write to pretty log file with colors
        with open(pretty_log_file, "a") as f:
            f.write(f"{BOLD}{'=' * 120}{RESET}\n")
            f.write(f"{BOLD}Timestamp: {timestamp}{RESET}\n")
            f.write(f"{BOLD}{'=' * 120}{RESET}\n\n")

            for message in converted_history:
                role = message.get("role", "unknown")

                if role == "system":
                    formatted = format_message_pretty(message, YELLOW, "system")
                elif role == "user":
                    formatted = format_message_pretty(message, MAGENTA, "user")
                elif role == "assistant":
                    formatted = format_message_pretty(message, CYAN, "assistant")
                else:
                    formatted = format_message_pretty(message, RESET, role)

                f.write(formatted + "\n")

            f.write("\n\n")

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
