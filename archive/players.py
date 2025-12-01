from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Callable


class Role(Enum):
    CREWMATE = "Crewmate"
    IMPOSTOR = "Impostor"
    MASTER_IMPOSTOR = "Master Impostor"


@dataclass
class LLMPlayer:
    name: str
    llm_fn: Callable
    role: Role = None
    sees: list[str] = field(default_factory=list)


# --- Create 5 LLM players ---
def setup_5_players(llm_functions: list[Callable]):
    """
    llm_functions: a list of 5 callables (each is an LLM interface).
    Returns: list of LLMPlayer with roles assigned + secret info.
    """

    players = [
        LLMPlayer(name=f"P{i+1}", llm_fn=llm) for i, llm in enumerate(llm_functions)
    ]

    deck = [Role.CREWMATE, Role.CREWMATE, Role.CREWMATE, Role.IMPOSTOR, Role.MASTER_IMPOSTOR]
    random.shuffle(deck)

    for p, r in zip(players, deck):
        p.role = r

    # Assign vision rules:
    # • Impostor knows other Impostor AND Master Impostor (in 5p there is 1 F + 1 H)
    # • Master Impostor **knows the Impostor in 5p**
    # • Crewmates know nothing
    impostor = [p for p in players if p.role == Role.IMPOSTOR]
    master_impostor = [p for p in players if p.role == Role.MASTER_IMPOSTOR][0]

    # Impostor sees Master Impostor
    if impostor:
        impostor_player = impostor[0]
        impostor_player.sees = [master_impostor.name]

    # Master Impostor sees Impostor (only in 5–6p)
    master_impostor.sees = [impostor[0].name] if impostor else []

    return players
