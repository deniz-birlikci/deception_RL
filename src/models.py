from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal, Annotated, TypeVar, Generic


class AIModel(str, Enum):
    OPENROUTER_GLM_4_5V = "z-ai/glm-4.5v"
    OPENROUTER_QWEN3_VL_235B_A22B_INSTRUCT = "qwen/qwen3-vl-235b-a22b-instruct"
    OPENROUTER_CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    OPENROUTER_GROK_4_FAST = "x-ai/grok-4-fast"
    OPENROUTER_GPT_5_CODEX = "openai/gpt-5-codex"
    OPENROUTER_GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"
    OPENROUTER_LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"
    OPENROUTER_GPT_5 = "openai/gpt-5"
    OPENROUTER_GPT_5_MINI = "openai/gpt-5-mini"
    OPENROUTER_GPT_4_1 = "openai/gpt-4.1"
    OPENROUTER_GPT_4o = "openai/gpt-4o"
    OPENROUTER_CLAUDE_3_5_HAIKU = "anthropic/claude-3.5-haiku"
    OPENROUTER_GEMINI_2_5_PRO = "google/gemini-2.5-pro"


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
