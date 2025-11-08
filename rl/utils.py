from typing import Any
from pydantic import BaseModel


class TerminalState(BaseModel):
    game_id: str
    reward: float
    # TODO: implement the game state to be returned here to
    # we want to save it to WandB afterwards


class ModelInput(BaseModel):
    messages: list[dict[str, Any]]
    terminal_state: TerminalState | None
