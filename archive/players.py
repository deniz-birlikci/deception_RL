from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Callable


class Role(Enum):
    LIBERAL = "Liberal"
    FASCIST = "Fascist"
    HITLER = "Hitler"


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

    deck = [Role.LIBERAL, Role.LIBERAL, Role.LIBERAL, Role.FASCIST, Role.HITLER]
    random.shuffle(deck)

    for p, r in zip(players, deck):
        p.role = r

    # Assign vision rules:
    # • Fascist knows other Fascist AND Hitler (in 5p there is 1 F + 1 H)
    # • Hitler **knows the Fascist in 5p**
    # • Liberals know nothing
    fascist = [p for p in players if p.role == Role.FASCIST]
    hitler = [p for p in players if p.role == Role.HITLER][0]

    # Fascist sees Hitler
    if fascist:
        f = fascist[0]
        f.sees = [hitler.name]

    # Hitler sees Fascist (only in 5–6p)
    hitler.sees = [fascist[0].name] if fascist else []

    return players
