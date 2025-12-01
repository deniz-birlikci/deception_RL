from typing import Any
from pydantic import BaseModel, Field
from src.models import Agent


class TerminalState(BaseModel):
    """
    Represents the final state of a completed game.

    Attributes:
        game_id: Unique identifier for the game
        winners: List of agents who won the game
        reward: Final reward value (based on win/loss)
        winning_team: Which team won the game ("crewmate" or "impostor")
    """

    game_id: str
    winners: list[Agent]
    reward: float
    winning_team: str | None = None
    trainable_agent_id: str | None = None
    emdash_counts: dict[str, int] = Field(default_factory=dict)


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
    Add a required 'reasoning' field to a tool schema as the FIRST property.
    This ensures the model generates reasoning before other parameters.
    """
    import copy
    
    schema = copy.deepcopy(schema)
    
    if "function" in schema and "parameters" in schema["function"]:
        params = schema["function"]["parameters"]
        
        # Get existing properties (or create empty dict)
        existing_properties = params.get("properties", {})
        
        # Create new properties dict with reasoning FIRST
        new_properties = {
            "reasoning": {
                "type": "string",
                "description": "Explain your reasoning behind the action you are taking. Think step-by-step about why this is the right choice."
            }
        }
        
        # Add existing properties after reasoning
        for key, value in existing_properties.items():
            if key != "reasoning":  # Don't duplicate if already exists
                new_properties[key] = value
        
        params["properties"] = new_properties
        
        # Add to required fields
        if "required" not in params:
            params["required"] = []
        
        if "reasoning" not in params["required"]:
            params["required"].insert(0, "reasoning")  # Insert at beginning
    
    return schema
