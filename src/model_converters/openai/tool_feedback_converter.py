from ...models import ToolFeedback, ToolResult, AIModel
from ..model_converter_protocols import DictConverter
from .tool_result_converter import OpenAIToolResultConverter
from ...env import settings


class OpenAIToolFeedbackConverter:

    def __init__(
        self,
        ai_model: AIModel | None = None,
        tool_result_converter: DictConverter[ToolResult] | None = None,
    ) -> None:
        self.ai_model = ai_model or settings.ai.model_id
        self.tool_result_converter = (
            tool_result_converter or OpenAIToolResultConverter()
        )

    def to_list_dict(self, data: ToolFeedback) -> list[dict]:
        tool_call_messages = [
            self.tool_result_converter.to_dict(tr) for tr in data.tool_call_results
        ]

        return tool_call_messages

    def from_dict(self, data: dict) -> ToolFeedback:
        raise NotImplementedError(
            f"openai format does not support deserializing ToolFeedback from dict: {data}"
        )
