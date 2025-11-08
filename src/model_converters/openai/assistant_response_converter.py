import time
from ...models import (
    AIModel,
    AssistantResponse,
    ToolCall,
    Tools,
    PresidentPickChancellorTool,
    VoteChancellorYesNoTool,
    PresidentChooseCardToDiscardTool,
    ChancellorPlayPolicyTool,
    ChooseAgentToVoteOutTool,
    AskAgentIfWantsToSpeakTool,
    AgentResponseToQuestionTool,
)
from ..model_converter_protocols import DictConverter
from .tool_call_converter import OpenAIToolCallConverter
from ..generic.president_pick_chancellor_tool_converter import (
    PresidentPickChancellorToolConverter,
)
from ..generic.vote_chancellor_yes_no_tool_converter import (
    VoteChancellorYesNoToolConverter,
)
from ..generic.president_choose_card_to_discard_tool_converter import (
    PresidentChooseCardToDiscardToolConverter,
)
from ..generic.chancellor_play_policy_tool_converter import (
    ChancellorPlayPolicyToolConverter,
)
from ..generic.choose_agent_to_vote_out_tool_converter import (
    ChooseAgentToVoteOutToolConverter,
)
from ..generic.ask_agent_if_wants_to_speak_tool_converter import (
    AskAgentIfWantsToSpeakToolConverter,
)
from ..generic.agent_response_to_question_tool_converter import (
    AgentResponseToQuestionToolConverter,
)
from ...env import settings


class OpenAIAssistantResponseConverter:

    def __init__(
        self,
        ai_model: AIModel | None = None,
        tool_call_converter: DictConverter[ToolCall] | None = None,
        president_pick_chancellor_tool_converter: (
            DictConverter[PresidentPickChancellorTool] | None
        ) = None,
        vote_chancellor_yes_no_tool_converter: (
            DictConverter[VoteChancellorYesNoTool] | None
        ) = None,
        president_choose_card_to_discard_tool_converter: (
            DictConverter[PresidentChooseCardToDiscardTool] | None
        ) = None,
        chancellor_play_policy_tool_converter: (
            DictConverter[ChancellorPlayPolicyTool] | None
        ) = None,
        choose_agent_to_vote_out_tool_converter: (
            DictConverter[ChooseAgentToVoteOutTool] | None
        ) = None,
        ask_agent_if_wants_to_speak_tool_converter: (
            DictConverter[AskAgentIfWantsToSpeakTool] | None
        ) = None,
        agent_response_to_question_tool_converter: (
            DictConverter[AgentResponseToQuestionTool] | None
        ) = None,
    ) -> None:
        self.ai_model = ai_model or settings.ai.model_id

        self.tool_call_converter = tool_call_converter or OpenAIToolCallConverter()
        self.president_pick_chancellor_tool_converter = (
            president_pick_chancellor_tool_converter
            or PresidentPickChancellorToolConverter()
        )
        self.vote_chancellor_yes_no_tool_converter = (
            vote_chancellor_yes_no_tool_converter or VoteChancellorYesNoToolConverter()
        )
        self.president_choose_card_to_discard_tool_converter = (
            president_choose_card_to_discard_tool_converter
            or PresidentChooseCardToDiscardToolConverter()
        )
        self.chancellor_play_policy_tool_converter = (
            chancellor_play_policy_tool_converter or ChancellorPlayPolicyToolConverter()
        )
        self.choose_agent_to_vote_out_tool_converter = (
            choose_agent_to_vote_out_tool_converter
            or ChooseAgentToVoteOutToolConverter()
        )
        self.ask_agent_if_wants_to_speak_tool_converter = (
            ask_agent_if_wants_to_speak_tool_converter
            or AskAgentIfWantsToSpeakToolConverter()
        )
        self.agent_response_to_question_tool_converter = (
            agent_response_to_question_tool_converter
            or AgentResponseToQuestionToolConverter()
        )

    def to_list_dict(self, data: AssistantResponse) -> list[dict]:
        assistant_content = [
            {"type": "text", "text": data.text_response},
        ]

        tool_call_parts = [
            self.tool_call_converter.to_dict(tc) for tc in data.tool_calls
        ]

        assistant_message = {
            "role": "assistant",
            "tool_calls": tool_call_parts,
            "content": assistant_content,
        }

        return [assistant_message]

    def from_dict(self, data: dict) -> AssistantResponse:
        choice_0_message = data.get("choices", [])[0].get("message", {})
        tool_calls_raw = choice_0_message.get("tool_calls", []) or []
        tool_calls = [self.tool_call_converter.from_dict(tc) for tc in tool_calls_raw]
        hydrated_tool_calls = [self._get_hydrated_tool(tc) for tc in tool_calls]
        return AssistantResponse(
            history_type="assistant-response",
            reasoning=choice_0_message.get("reasoning", None),
            text_response=choice_0_message.get("content", None),
            tool_calls=tool_calls,
            hydrated_tool_calls=hydrated_tool_calls,
            timestamp=str(int(time.time() * 1000)),
        )

    def _get_hydrated_tool(self, tool_call: ToolCall) -> Tools:
        if tool_call.tool_name == "president-pick-chancellor":
            return self.president_pick_chancellor_tool_converter.from_dict(
                tool_call.input
            )
        elif tool_call.tool_name == "vote-chancellor-yes-no":
            return self.vote_chancellor_yes_no_tool_converter.from_dict(tool_call.input)
        elif tool_call.tool_name == "president-choose-card-to-discard":
            return self.president_choose_card_to_discard_tool_converter.from_dict(
                tool_call.input
            )
        elif tool_call.tool_name == "chancellor-play-policy":
            return self.chancellor_play_policy_tool_converter.from_dict(tool_call.input)
        elif tool_call.tool_name == "choose-agent-to-vote-out":
            return self.choose_agent_to_vote_out_tool_converter.from_dict(
                tool_call.input
            )
        elif tool_call.tool_name == "ask-agent-if-wants-to-speak":
            return self.ask_agent_if_wants_to_speak_tool_converter.from_dict(
                tool_call.input
            )
        elif tool_call.tool_name == "agent-response-to-question-tool":
            return self.agent_response_to_question_tool_converter.from_dict(
                tool_call.input
            )
        else:
            raise ValueError(f"Unknown tool name: {tool_call.tool_name}")
