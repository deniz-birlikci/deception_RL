from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal, Annotated, TypeVar, Generic


class AIModel(str, Enum):
    OPENAI_GPT_5 = "gpt-5-2025-08-07"
    OPENAI_GPT_5_MINI = "gpt-5-mini-2025-08-07"
    OPENAI_GPT_5_NANO = "gpt-5-nano-2025-08-07"
    OPENAI_GPT_4_1 = "gpt-4.1-2025-04-14"


class Backend(str, Enum):
    OPENAI = "openai"


_TTool = TypeVar("_TTool", bound=str)


class _ToolBase(BaseModel, Generic[_TTool]):
    tool_type: _TTool


class PresidentPickChancellorTool(_ToolBase[Literal["president-pick-chancellor"]]):
    tool_type: Literal["president-pick-chancellor"]
    agent_id: str


class VoteChancellorYesNoTool(_ToolBase[Literal["vote-chancellor-yes-no"]]):
    tool_type: Literal["vote-chancellor-yes-no"]
    choice: bool


class PresidentChooseCardToDiscardTool(
    _ToolBase[Literal["president-choose-card-to-discard"]]
):
    tool_type: Literal["president-choose-card-to-discard"]
    card_index: int


class ChancellorPlayPolicyTool(_ToolBase[Literal["chancellor-play-policy"]]):
    tool_type: Literal["chancellor-play-policy"]
    card_index: int


class ChooseAgentToVoteOutTool(_ToolBase[Literal["choose-agent-to-vote-out"]]):
    tool_type: Literal["choose-agent-to-vote-out"]
    agent_id: str | None


class AskAgentIfWantsToSpeakTool(_ToolBase[Literal["ask-agent-if-wants-to-speak"]]):
    tool_type: Literal["ask-agent-if-wants-to-speak"]
    question_or_statement: str | None
    ask_directed_question_to_agent_id: str | None


class AgentResponseToQuestionTool(
    _ToolBase[Literal["agent-response-to-question-tool"]]
):
    tool_type: Literal["agent-response-to-question-tool"]
    response: str


Tools = Annotated[
    PresidentPickChancellorTool
    | VoteChancellorYesNoTool
    | PresidentChooseCardToDiscardTool
    | ChancellorPlayPolicyTool
    | ChooseAgentToVoteOutTool
    | AskAgentIfWantsToSpeakTool
    | AgentResponseToQuestionTool,
    Field(discriminator="tool_type"),
]


class _ToolCallBase(BaseModel):
    tool_call_id: str
    tool_name: str


class ToolCall(_ToolCallBase):
    input: dict


class ToolResult(_ToolCallBase):
    output: str


_THistory = TypeVar("_THistory", bound=str)


class _MessageHistoryBase(BaseModel, Generic[_THistory]):
    history_type: _THistory
    timestamp: str


class UserInput(
    _MessageHistoryBase[Literal["user-input"]],
):
    history_type: Literal["user-input"]
    user_message: str


class ToolFeedback(
    _MessageHistoryBase[Literal["tool-feedback"]],
):
    history_type: Literal["tool-feedback"]
    tool_call_results: list[ToolResult]


class AssistantResponse(
    _MessageHistoryBase[Literal["assistant-response"]],
):
    history_type: Literal["assistant-response"]
    reasoning: str | None = None
    text_response: str | None = None
    tool_calls: list[ToolCall]
    hydrated_tool_calls: list[Tools]


MessageHistory = Annotated[
    ToolFeedback | UserInput | AssistantResponse,
    Field(discriminator="history_type"),
]


class AgentRole(str, Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"
    HITLER = "hitler"


class Agent(BaseModel):
    agent_id: str
    role: AgentRole
    ai_model: AIModel | None = None


class PolicyCard(str, Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"


class PresidentPickChancellorEventPublic(BaseModel):
    president_id: str
    chancellor_id: str


class VoteChancellorYesNoEventPublic(BaseModel):
    voter_id: str
    chancellor_nominee_id: str
    vote: bool


class ChooseAgentToVoteOutEventPublic(BaseModel):
    voter_id: str
    nominated_agent_id: str | None


class AskAgentIfWantsToSpeakEventPublic(BaseModel):
    agent_id: str
    question_or_statement: str | None
    ask_directed_question_to_agent_id: str | None


class AgentResponseToQuestioningEventPublic(BaseModel):
    agent_id: str
    in_response_to_agent_id: str
    response: str


class PresidentChooseCardToDiscardEventPrivate(BaseModel):
    president_id: str
    cards_drawn: list[PolicyCard]
    card_discarded: PolicyCard


class ChancellorReceivePoliciesEventPrivate(BaseModel):
    chancellor_id: str
    president_id_received_from: str
    cards_received: list[PolicyCard]
    card_discarded: PolicyCard


class ChancellorPlayPolicyEventPublic(BaseModel):
    chancellor_id: str
    card_played: PolicyCard
