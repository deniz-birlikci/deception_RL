from typing import Any
from pydantic import BaseModel


class TerminalState(BaseModel):
    """
    Represents the final state of a completed game.

    Attributes:
        game_id: Unique identifier for the game
        reward: Final reward value (based on win/loss)
        winner: Which team won ('crewmate' or 'impostor')
        security_protocols_resolved: Number of security protocols completed
        sabotage_protocols_completed: Number of sabotage protocols completed
        master_impostor_elected: Whether the Master Impostor was promoted after enough sabotages
        num_executions: Number of players executed during the game
    """

    game_id: str
    reward: float
    winner: str | None = None
    security_protocols_resolved: int = 0
    sabotage_protocols_completed: int = 0
    master_impostor_elected: bool = False
    num_executions: int = 0


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


# TODO: These functions will be implemented by your teammate working on the engine
# For now, we provide stubs that the engine team can fill in


async def create_game() -> ModelInput:
    """
    Create a new Secret Impostor game and return the initial game state.

    This function should:
    1. Initialize a new game with 5 agents
    2. Assign roles (Crewmate, Impostor, Master Impostor)
    3. Determine the first president
    4. Build initial messages for the first agent to act
    5. Return ModelInput with initial state

    Returns:
        ModelInput with initial messages and terminal_state=None

    Example messages format:
    [
        {
            "role": "system",
            "content": "You are Agent <agent_id> playing Secret Impostor. Your role is <role>. ..."
        },
        {
            "role": "user",
            "content": "Game State:\\nCrewmate Policies: 0\\nImpostor Policies: 0\\n...\\n\\nYou are the President. Choose a Chancellor."
        }
    ]
    """
    # TODO: Implement this function
    # For now, raise an error to indicate it needs implementation
    raise NotImplementedError(
        "create_game() must be implemented by the engine team. "
        "It should initialize a game and return the first ModelInput."
    )


async def execute_action(tool_call_json: str) -> ModelInput:
    """
    Execute an action in the game and return the next state.

    This function should:
    1. Parse the tool_call_json to extract tool name and arguments
    2. Validate the action is legal in current game state
    3. Execute the action (update game state)
    4. Determine the next agent to act
    5. Build messages for that agent
    6. Check if game is over (win condition met)
    7. Return ModelInput with new state

    Args:
        tool_call_json: JSON string like:
            {
                "tool_name": "president-pick-chancellor",
                "arguments": {"agent_id": "abc-123"}
            }

    Returns:
        ModelInput with updated messages and possibly terminal_state

    Example flow:
    - If game continues: return ModelInput(messages=[...], terminal_state=None)
    - If game ends: return ModelInput(
          messages=[],  # Can be empty since game is over
          terminal_state=TerminalState(
              game_id="...",
              reward=1.0,  # +1 for crewmate win, -1 for impostor win
              winner="crewmate",
              ...
          )
      )
    """
    # TODO: Implement this function
    # For now, raise an error to indicate it needs implementation
    raise NotImplementedError(
        "execute_action() must be implemented by the engine team. "
        "It should execute the action and return the next ModelInput."
    )
