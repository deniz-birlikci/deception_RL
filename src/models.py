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


Tools = Annotated[
    PresidentPickChancellorTool
    | VoteChancellorYesNoTool
    | PresidentChooseCardToDiscardTool
    | ChancellorPlayPolicyTool
    | ChooseAgentToVoteOutTool,
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
