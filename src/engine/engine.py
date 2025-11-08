import random
from src.models import (
    Agent,
    AgentRole,
    PresidentPickChancellorEventPublic,
    VoteChancellorYesNoEventPublic,
    ChooseAgentToVoteOutEventPublic,
    PresidentChooseCardToDiscardEventPrivate,
    ChancellorPlayPolicyEventPrivate,
)
from src.engine.deck import Deck


class Engine:
    def __init__(
        self,
        deck: Deck,
        fascist_policies_to_win: int = 6,
        liberal_policies_to_win: int = 5,
    ) -> None:
        self.deck = deck
        self.fascist_policies_to_win = fascist_policies_to_win
        self.liberal_policies_to_win = liberal_policies_to_win
        self.hitler_election_threshold = fascist_policies_to_win // 2

        self.agents: list[Agent] = []
        self.president_rotation: list[str] = []
        self.current_president_id: str | None = None
        self.current_chancellor_id: str | None = None
        self.fascist_policies_played: int = 0
        self.liberal_policies_played: int = 0

        self.public_events: list[
            PresidentPickChancellorEventPublic
            | VoteChancellorYesNoEventPublic
            | ChooseAgentToVoteOutEventPublic
        ] = []

        self.private_events_by_agent: dict[
            str,
            list[
                PresidentChooseCardToDiscardEventPrivate
                | ChancellorPlayPolicyEventPrivate
            ],
        ] = {}

    def create(self, agent_ids: list[str]) -> None:
        roles = [
            AgentRole.HITLER,
            AgentRole.FASCIST,
            AgentRole.LIBERAL,
            AgentRole.LIBERAL,
            AgentRole.LIBERAL,
        ]

        shuffled_roles = random.sample(roles, len(roles))

        self.agents = [
            Agent(agent_id=agent_id, role=role)
            for agent_id, role in zip(agent_ids, shuffled_roles)
        ]

        self.president_rotation = random.sample(agent_ids, len(agent_ids))

        self.current_president_id = random.choice(agent_ids)
        self.current_chancellor_id = None

        for agent_id in agent_ids:
            self.private_events_by_agent[agent_id] = []
