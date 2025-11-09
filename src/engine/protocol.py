from typing import Any
from pydantic import BaseModel
from src.models import Agent


class TerminalState(BaseModel):
    """
    Represents the final state of a completed game.

    Attributes:
        reward: Final reward value (based on win/loss)
    """

    # TODO(Charlie): let's define this together
    winners: list[Agent]
    reward: float


class ToolCallTarget(BaseModel):
    name: str
    openai_schema: dict[str, Any]


class ModelInput(BaseModel):
    """
    Input provided to the model at each turn.

    Attributes:
        messages: List of messages in OpenAI format (system, user, assistant, tool)
        terminal_state: If not None, the game is over
    """

    messages: list[dict[str, Any]]
    tool_call: ToolCallTarget | None
    terminal_state: TerminalState | None


class ModelOutput(BaseModel):
    # JSON string for the function calling reslt
    function_calling_json: str
    reasoning: str | None = None
