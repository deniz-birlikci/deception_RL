import random
import uuid
from queue import Queue
from typing import cast
import json
from src.models import (
    Agent,
    AgentRole,
    AIModel,
    PolicyCard,
    Backend,
    PresidentPickChancellorEventPublic,
    VoteChancellorYesNoEventPublic,
    AskAgentIfWantsToSpeakEventPublic,
    AgentResponseToQuestioningEventPublic,
    PresidentChooseCardToDiscardEventPrivate,
    ChancellorReceivePoliciesEventPrivate,
    ChancellorPlayPolicyEventPublic,
    PresidentPickChancellorTool,
    VoteChancellorYesNoTool,
    PresidentChooseCardToDiscardTool,
    ChancellorPlayPolicyTool,
    AskAgentIfWantsToSpeakTool,
    AgentResponseToQuestionTool,
    UserInput,
    ToolFeedback,
    ToolResult,
)
from src.agent import BaseAgent
from src.engine.deck import Deck
from src.agent.agent_registry import AgentRegistry
from src.engine.external_agent_response_parser import ExternalAgentResponseParser
from src.models import Tools

ROLES = [
    AgentRole.HITLER,
    AgentRole.FASCIST,
    AgentRole.LIBERAL,
    AgentRole.LIBERAL,
    AgentRole.LIBERAL,
]


class Engine:
    def __init__(
        self,
        deck: Deck,
        ai_models: list[AIModel | None],
        fascist_policies_to_win: int,
        liberal_policies_to_win: int,
        log_file: str | None = None,
    ) -> None:
        self.deck = deck
        self.fascist_policies_to_win = fascist_policies_to_win
        self.liberal_policies_to_win = liberal_policies_to_win
        self.log_file = log_file

        shuffled_roles = random.sample(ROLES, len(ROLES))
        agent_ids = [f"agent_{i}" for i in range(len(ai_models))]
        agents = [
            Agent(agent_id=aid, role=role, ai_model=model)
            for aid, role, model in zip(agent_ids, shuffled_roles, ai_models)
        ]

        self.agents_by_id: dict[str, Agent] = {a.agent_id: a for a in agents}
        self.president_rotation: list[str] = random.sample(agent_ids, len(agent_ids))
        self.current_president_idx: int = 0
        self.current_chancellor_id: str | None = None
        self.fascist_policies_played: int = 0
        self.liberal_policies_played: int = 0
        self.failed_election_tracker: int = 0
        self.public_events: list = []
        self.private_events_by_agent: dict[str, list] = {aid: [] for aid in agent_ids}

        self.ai_agents: dict[str, BaseAgent] = {}
        self.msg_history: dict[str, list] = {aid: [] for aid in agent_ids}
        for aid, agent in self.agents_by_id.items():
            if agent.ai_model:
                self.ai_agents[aid] = AgentRegistry.create_agent(
                    backend=Backend.OPENAI, ai_model=agent.ai_model
                )

    def run(self, input_queue: Queue, output_queue: Queue) -> None:
        while not self._is_game_over():
            president_id = self.president_rotation[self.current_president_idx]

            tool = cast(
                PresidentPickChancellorTool,
                self._get_tool(
                    president_id,
                    "Nominate a Chancellor.",
                    input_queue,
                    output_queue,
                    allowed_tools=["president-pick-chancellor"],
                ),
            )
            chancellor_id = tool.agent_id
            self.public_events.append(
                PresidentPickChancellorEventPublic(
                    president_id=president_id, chancellor_id=chancellor_id
                )
            )
            self._log_state_to_file()

            self._discourse(input_queue, output_queue)
            self._log_state_to_file()

            if not self._vote(chancellor_id, input_queue, output_queue):
                self.failed_election_tracker += 1
                self._log_state_to_file()

                if self.failed_election_tracker >= 3:
                    self._handle_failed_election()
                    self._log_state_to_file()

                self.current_president_idx = (self.current_president_idx + 1) % len(
                    self.president_rotation
                )
                self._log_state_to_file()
                continue

            self.failed_election_tracker = 0
            self.current_chancellor_id = chancellor_id
            self._log_state_to_file()

            cards = self.deck.draw(3)
            tool = cast(
                PresidentChooseCardToDiscardTool,
                self._get_tool(
                    president_id,
                    f"Cards: {cards}. Discard index (0-2).",
                    input_queue,
                    output_queue,
                    allowed_tools=["president-choose-card-to-discard"],
                ),
            )
            idx = tool.card_index
            discarded = cards.pop(idx)
            self.deck.add_to_discard(discarded)

            self.private_events_by_agent[president_id].append(
                PresidentChooseCardToDiscardEventPrivate(
                    president_id=president_id,
                    cards_drawn=cards + [discarded],
                    card_discarded=discarded,
                )
            )
            self._log_state_to_file()

            tool = cast(
                ChancellorPlayPolicyTool,
                self._get_tool(
                    chancellor_id,
                    f"Cards: {cards}. Play index (0-1).",
                    input_queue,
                    output_queue,
                    allowed_tools=["chancellor-play-policy"],
                ),
            )
            idx = tool.card_index
            played = cards[idx]
            discarded_by_chancellor = cards[1 - idx]
            self.deck.add_to_discard(discarded_by_chancellor)
            
            self.private_events_by_agent[chancellor_id].append(
                ChancellorReceivePoliciesEventPrivate(
                    chancellor_id=chancellor_id,
                    president_id_received_from=president_id,
                    cards_received=cards,
                    card_discarded=discarded_by_chancellor,
                )
            )
            self.public_events.append(
                ChancellorPlayPolicyEventPublic(
                    chancellor_id=chancellor_id, card_played=played
                )
            )
            self._log_state_to_file()

            if played == PolicyCard.FASCIST:
                self.fascist_policies_played += 1
            else:
                self.liberal_policies_played += 1

            self._discourse(input_queue, output_queue)
            self._log_state_to_file()

            self.current_president_idx = (self.current_president_idx + 1) % len(
                self.president_rotation
            )
            self._log_state_to_file()

        output_queue.put(self._get_winners())

    def _get_tool(
        self,
        agent_id: str,
        prompt_guidance: str,
        input_queue: Queue,
        output_queue: Queue,
        allowed_tools: list[str] | None = None,
    ) -> Tools:
        agent = self.agents_by_id[agent_id]
        full_prompt = self._build_prompt_for_agent(agent_id, prompt_guidance)

        if agent.ai_model is None:
            output_queue.put(full_prompt)
            response = ExternalAgentResponseParser.parse(input_queue.get())
        else:
            user_input = UserInput(
                history_type="user-input",
                user_message=full_prompt,
                timestamp=str(uuid.uuid4()),
            )
            self.msg_history[agent_id].append(user_input)
            response = self.ai_agents[agent_id].generate_response(
                self.msg_history[agent_id], allowed_tools=allowed_tools
            )
            self.msg_history[agent_id].append(response)

        feedback = ToolFeedback(
            history_type="tool-feedback",
            tool_call_results=[
                ToolResult(
                    tool_call_id=response.tool_calls[0].tool_call_id,
                    tool_name=response.tool_calls[0].tool_name,
                    output="OK",
                )
            ],
            timestamp=str(uuid.uuid4()),
        )
        self.msg_history[agent_id].append(feedback)

        return response.hydrated_tool_calls[0]

    def _discourse(self, input_queue: Queue, output_queue: Queue) -> None:
        speakers = []
        for aid in self.agents_by_id:
            tool = cast(
                AskAgentIfWantsToSpeakTool,
                self._get_tool(
                    aid,
                    "Speak? (question/statement or null)",
                    input_queue,
                    output_queue,
                    allowed_tools=["ask-agent-if-wants-to-speak"],
                ),
            )
            if tool.question_or_statement:
                speakers.append((aid, tool))

        random.shuffle(speakers)

        for aid, tool in speakers:
            self.public_events.append(
                AskAgentIfWantsToSpeakEventPublic(
                    agent_id=aid,
                    question_or_statement=tool.question_or_statement,
                    ask_directed_question_to_agent_id=tool.ask_directed_question_to_agent_id,
                )
            )

            if tool.ask_directed_question_to_agent_id:
                target = tool.ask_directed_question_to_agent_id
                resp = cast(
                    AgentResponseToQuestionTool,
                    self._get_tool(
                        target,
                        f"Asked: {tool.question_or_statement}. Respond.",
                        input_queue,
                        output_queue,
                        allowed_tools=["agent-response-to-question-tool"],
                    ),
                )
                self.public_events.append(
                    AgentResponseToQuestioningEventPublic(
                        agent_id=target,
                        in_response_to_agent_id=aid,
                        response=resp.response,
                    )
                )

    def _handle_failed_election(self) -> None:
        top_card = self.deck.draw(1)[0]
        self.public_events.append(
            ChancellorPlayPolicyEventPublic(chancellor_id=None, card_played=top_card)
        )
        if top_card == PolicyCard.FASCIST:
            self.fascist_policies_played += 1
        else:
            self.liberal_policies_played += 1
        self.failed_election_tracker = 0

    def _vote(
        self, chancellor_id: str, input_queue: Queue, output_queue: Queue
    ) -> bool:
        votes = []
        for aid in self.agents_by_id:
            tool = cast(
                VoteChancellorYesNoTool,
                self._get_tool(
                    aid,
                    f"Vote on Chancellor {chancellor_id}? (true/false)",
                    input_queue,
                    output_queue,
                    allowed_tools=["vote-chancellor-yes-no"],
                ),
            )
            votes.append(tool.choice)
            self.public_events.append(
                VoteChancellorYesNoEventPublic(
                    voter_id=aid, chancellor_nominee_id=chancellor_id, vote=tool.choice
                )
            )
        return sum(votes) > len(votes) // 2

    def _is_game_over(self) -> bool:
        return (
            self.fascist_policies_played >= self.fascist_policies_to_win
            or self.liberal_policies_played >= self.liberal_policies_to_win
            or self._hitler_elected()
        )

    def _hitler_elected(self) -> bool:
        if self.fascist_policies_played < 3:
            return False
        if self.current_chancellor_id is None:
            return False
        return self.agents_by_id[self.current_chancellor_id].role == AgentRole.HITLER

    def _get_winners(self) -> list[Agent]:
        if (
            self.fascist_policies_played >= self.fascist_policies_to_win
            or self._hitler_elected()
        ):
            return [
                agent
                for agent in self.agents_by_id.values()
                if agent.role in [AgentRole.HITLER, AgentRole.FASCIST]
            ]
        else:
            return [
                agent
                for agent in self.agents_by_id.values()
                if agent.role == AgentRole.LIBERAL
            ]

    def _build_prompt_for_agent(self, agent_id: str, action_prompt: str) -> str:
        agent = self.agents_by_id[agent_id]

        game_state = f"""=== GAME STATE ===
        Your Agent ID: {agent_id}
        Your Role: {agent.role}
        Current President: {self.president_rotation[self.current_president_idx]}
        Current Chancellor: {self.current_chancellor_id if self.current_chancellor_id else 'None'}
        Fascist Policies Played: {self.fascist_policies_played}/{self.fascist_policies_to_win}
        Liberal Policies Played: {self.liberal_policies_played}/{self.liberal_policies_to_win}
        Failed Elections: {self.failed_election_tracker}/3 (at 3, top policy is played automatically)

        All Agents: {list(self.agents_by_id.keys())}
        """

        public_events_str = "\n=== PUBLIC EVENTS ===\n"
        if self.public_events:
            for i, event in enumerate(self.public_events, 1):
                public_events_str += f"{i}. {event}\n"
        else:
            public_events_str += "No public events yet.\n"

        private_events_str = "\n=== YOUR PRIVATE EVENTS ===\n"
        if self.private_events_by_agent[agent_id]:
            for i, event in enumerate(self.private_events_by_agent[agent_id], 1):
                private_events_str += f"{i}. {event}\n"
        else:
            private_events_str += "No private events yet.\n"

        fascist_info_str = ""
        fascist_ids = [
            aid
            for aid, a in self.agents_by_id.items()
            if a.role in [AgentRole.FASCIST, AgentRole.HITLER] and aid != agent_id
        ]
        if agent.role == AgentRole.FASCIST:
            fascist_info_str = f"\n=== FELLOW FASCISTS ===\n{fascist_ids}\n"

        action_str = f"\n=== ACTION REQUIRED ===\n{action_prompt}\n"
        return (
            game_state
            + public_events_str
            + private_events_str
            + fascist_info_str
            + action_str
        )

    def _log_state_to_file(self) -> None:
        if not self.log_file:
            return

        state = {
            "agents": {
                aid: {"role": agent.role.value, "ai_model": agent.ai_model}
                for aid, agent in self.agents_by_id.items()
            },
            "president_rotation": self.president_rotation,
            "current_president_idx": self.current_president_idx,
            "current_chancellor_id": self.current_chancellor_id,
            "fascist_policies_played": self.fascist_policies_played,
            "liberal_policies_played": self.liberal_policies_played,
            "failed_election_tracker": self.failed_election_tracker,
            "public_events": [str(e) for e in self.public_events],
            "private_events_by_agent": {
                aid: [str(e) for e in events]
                for aid, events in self.private_events_by_agent.items()
            },
        }

        with open(self.log_file, "a") as f:
            f.write("=" * 60 + "\n")
            f.write(json.dumps(state, indent=2) + "\n")
