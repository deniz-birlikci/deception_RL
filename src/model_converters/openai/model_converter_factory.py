from ...models import (
    Backend,
    ToolCall,
    ToolResult,
    UserInput,
    ToolFeedback,
    AssistantResponse,
)
from ..model_converter_protocols import DictConverter, ListDictConverter
from ..base_model_converter_factory import BaseModelConverterFactory
from .tool_call_converter import OpenAIToolCallConverter
from .tool_result_converter import OpenAIToolResultConverter
from .user_input_converter import OpenAIUserInputConverter
from .tool_feedback_converter import OpenAIToolFeedbackConverter
from .assistant_response_converter import OpenAIAssistantResponseConverter


class OpenAIModelConverterFactory(BaseModelConverterFactory):

    backend: Backend = Backend.OPENAI

    def create_tool_call_converter(self) -> DictConverter[ToolCall]:
        return OpenAIToolCallConverter()

    def create_tool_result_converter(self) -> DictConverter[ToolResult]:
        return OpenAIToolResultConverter()

    def create_user_input_converter(self) -> ListDictConverter[UserInput]:
        return OpenAIUserInputConverter(ai_model=self.ai_model)

    def create_tool_feedback_converter(self) -> ListDictConverter[ToolFeedback]:
        return OpenAIToolFeedbackConverter(ai_model=self.ai_model)

    def create_assistant_response_converter(
        self,
    ) -> ListDictConverter[AssistantResponse]:
        return OpenAIAssistantResponseConverter(ai_model=self.ai_model)
