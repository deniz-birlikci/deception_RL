from abc import ABC, abstractmethod
from ..models import (
    Backend,
    AIModel,
    ToolCall,
    ToolResult,
    UserInput,
    ToolFeedback,
    AssistantResponse,
    PresidentPickChancellorTool,
    VoteChancellorYesNoTool,
    PresidentChooseCardToDiscardTool,
    ChancellorPlayPolicyTool,
    ChooseAgentToVoteOutTool,
    AskAgentIfWantsToSpeakTool,
    AgentResponseToQuestionTool,
)
from .model_converter_protocols import DictConverter, ListDictConverter
from ..env import settings
from .generic.president_pick_chancellor_tool_converter import (
    PresidentPickChancellorToolConverter,
)
from .generic.vote_chancellor_yes_no_tool_converter import (
    VoteChancellorYesNoToolConverter,
)
from .generic.president_choose_card_to_discard_tool_converter import (
    PresidentChooseCardToDiscardToolConverter,
)
from .generic.chancellor_play_policy_tool_converter import (
    ChancellorPlayPolicyToolConverter,
)
from .generic.choose_agent_to_vote_out_tool_converter import (
    ChooseAgentToVoteOutToolConverter,
)
from .generic.ask_agent_if_wants_to_speak_tool_converter import (
    AskAgentIfWantsToSpeakToolConverter,
)
from .generic.agent_response_to_question_tool_converter import (
    AgentResponseToQuestionToolConverter,
)


class BaseModelConverterFactory(ABC):

    backend: Backend

    def __init__(
        self,
        ai_model: AIModel | None = None,
    ) -> None:
        self.ai_model = ai_model or settings.ai.model_id

    @abstractmethod
    def create_tool_call_converter(self) -> DictConverter[ToolCall]: ...

    @abstractmethod
    def create_tool_result_converter(self) -> DictConverter[ToolResult]: ...

    @abstractmethod
    def create_user_input_converter(self) -> ListDictConverter[UserInput]: ...

    @abstractmethod
    def create_tool_feedback_converter(self) -> ListDictConverter[ToolFeedback]: ...

    @abstractmethod
    def create_assistant_response_converter(
        self,
    ) -> ListDictConverter[AssistantResponse]: ...

    def create_president_pick_chancellor_tool_converter(
        self,
    ) -> DictConverter[PresidentPickChancellorTool]:
        return PresidentPickChancellorToolConverter()

    def create_vote_chancellor_yes_no_tool_converter(
        self,
    ) -> DictConverter[VoteChancellorYesNoTool]:
        return VoteChancellorYesNoToolConverter()

    def create_president_choose_card_to_discard_tool_converter(
        self,
    ) -> DictConverter[PresidentChooseCardToDiscardTool]:
        return PresidentChooseCardToDiscardToolConverter()

    def create_chancellor_play_policy_tool_converter(
        self,
    ) -> DictConverter[ChancellorPlayPolicyTool]:
        return ChancellorPlayPolicyToolConverter()

    def create_choose_agent_to_vote_out_tool_converter(
        self,
    ) -> DictConverter[ChooseAgentToVoteOutTool]:
        return ChooseAgentToVoteOutToolConverter()

    def create_ask_agent_if_wants_to_speak_tool_converter(
        self,
    ) -> DictConverter[AskAgentIfWantsToSpeakTool]:
        return AskAgentIfWantsToSpeakToolConverter()

    def create_agent_response_to_question_tool_converter(
        self,
    ) -> DictConverter[AgentResponseToQuestionTool]:
        return AgentResponseToQuestionToolConverter()
