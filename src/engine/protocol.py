from typing import Any
from pydantic import BaseModel
from src.models import Agent


class TerminalState(BaseModel):
    """
    Represents the final state of a completed game.

    Attributes:
        game_id: Unique identifier for the game
        winners: List of agents who won the game
        reward: Final reward value (based on win/loss)
    """

    game_id: str
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


def add_reasoning_to_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Add a required 'reasoning' field to a tool schema.
    """
    import copy
    
    schema = copy.deepcopy(schema)
    
    if "function" in schema and "parameters" in schema["function"]:
        params = schema["function"]["parameters"]
        
        # Add reasoning property
        if "properties" not in params:
            params["properties"] = {}
        
        params["properties"]["reasoning"] = {
            "type": "string",
            "description": "Give your reasoning behind the action that you are taking."
        }
        
        # Add to required fields
        if "required" not in params:
            params["required"] = []
        
        if "reasoning" not in params["required"]:
            params["required"].append("reasoning")
    
    return schema
