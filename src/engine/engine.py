import random
import time
import uuid
from src.models import (
    Agent,
    AgentRole,
    AIModel,
    PolicyCard,
    PresidentPickChancellorEventPublic,
    VoteChancellorYesNoEventPublic,
    ChooseAgentToVoteOutEventPublic,
    AskAgentIfWantsToSpeakEventPublic,
    AgentResponseToQuestioningEventPublic,
    PresidentChooseCardToDiscardEventPrivate,
    ChancellorReceivePoliciesEventPrivate,
    ChancellorPlayPolicyEventPublic,
    UserInput,
    PresidentPickChancellorTool,
    VoteChancellorYesNoTool,
    PresidentChooseCardToDiscardTool,
    ChancellorPlayPolicyTool,
    AskAgentIfWantsToSpeakTool,
    AgentResponseToQuestionTool,
)
from src.engine.deck import Deck
from src.agent.agent_registry import AgentRegistry
from src.env import settings


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

        self.agents_by_id: dict[str, Agent] = {}
        self.president_rotation: list[str] = []
        self.current_president_id: str | None = None
        self.current_chancellor_id: str | None = None
        self.fascist_policies_played: int = 0
        self.liberal_policies_played: int = 0

        self.public_events: list[
            PresidentPickChancellorEventPublic
            | VoteChancellorYesNoEventPublic
            | ChooseAgentToVoteOutEventPublic
            | AskAgentIfWantsToSpeakEventPublic
            | AgentResponseToQuestioningEventPublic
            | ChancellorPlayPolicyEventPublic
        ] = []

        self.private_events_by_agent: dict[
            str,
            list[
                PresidentChooseCardToDiscardEventPrivate
                | ChancellorReceivePoliciesEventPrivate
            ],
        ] = {}

    def create(
        self,
        ai_models: list[AIModel | None],
    ) -> None:
        roles = [
            AgentRole.HITLER,
            AgentRole.FASCIST,
            AgentRole.LIBERAL,
            AgentRole.LIBERAL,
            AgentRole.LIBERAL,
        ]

        shuffled_roles = random.sample(roles, len(roles))

        agent_ids = [str(uuid.uuid4()) for _ in range(len(ai_models))]

        agents = [
            Agent(agent_id=agent_id, role=role, ai_model=ai_model)
            for agent_id, role, ai_model in zip(agent_ids, shuffled_roles, ai_models)
        ]

        self.agents_by_id = {agent.agent_id: agent for agent in agents}

        for agent_id in agent_ids:
            self.private_events_by_agent[agent_id] = []

        self.president_rotation = random.sample(agent_ids, len(agent_ids))

        self.current_president_id = random.choice(agent_ids)
        self.current_chancellor_id = None

    