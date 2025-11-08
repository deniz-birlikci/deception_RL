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
    MessageHistory,
    UserInput,
    PresidentPickChancellorTool,
    VoteChancellorYesNoTool,
    PresidentChooseCardToDiscardTool,
    ChancellorPlayPolicyTool,
    AskAgentIfWantsToSpeakTool,
    AgentResponseToQuestionTool,
    Backend,
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

    def _now_ms(self) -> str:
        return str(int(time.time() * 1000))

    def _make_user_input(self, text: str) -> UserInput:
        return UserInput(history_type="user-input", timestamp=self._now_ms(), user_message=text)

    def _get_agent_backend(self) -> Backend:
        # Currently only OpenAI is wired.
        return Backend.OPENAI

    def _create_llm_agent_for(self, agent: Agent):
        return AgentRegistry.create_agent(
            backend=self._get_agent_backend(),
            ai_model=agent.ai_model,
        )

    def _prompt_agent_for_speak_intent(
        self,
        agent: Agent,
        other_agent_ids: list[str],
    ) -> AskAgentIfWantsToSpeakTool | None:
        """Ask an agent if they want to speak and optionally pose a directed question.

        Returns the hydrated AskAgentIfWantsToSpeakTool if they choose to speak, else None.
        """
        llm = self._create_llm_agent_for(agent)
        prompt = (
            "Discussion phase: If you want to speak, call the ask-agent-if-wants-to-speak tool. "
            "You may optionally include a 'question_or_statement' to say publicly, and/or set "
            "'ask_directed_question_to_agent_id' to one of these agents: "
            f"{other_agent_ids}. If you do not want to speak, do nothing."
        )
        history: list[MessageHistory] = [self._make_user_input(prompt)]
        assistant = llm.generate_response(history)
        for tool in assistant.hydrated_tool_calls:
            if isinstance(tool, AskAgentIfWantsToSpeakTool):
                # Consider any such tool call as intent to speak.
                return tool
        return None

    def _prompt_agent_for_direct_answer(
        self,
        target: Agent,
        asked_by_id: str,
        question_text: str | None,
    ) -> AgentResponseToQuestionTool | None:
        """Prompt a target agent to answer a directed question via the agent-response tool."""
        llm = self._create_llm_agent_for(target)
        prompt = (
            "You were asked a direct question in the discussion phase. "
            f"Question from agent {asked_by_id}: '{question_text or ''}'. "
            "Respond ONLY by calling the agent-response-to-question-tool with your 'response' string."
        )
        history: list[MessageHistory] = [self._make_user_input(prompt)]
        assistant = llm.generate_response(history)
        for tool in assistant.hydrated_tool_calls:
            if isinstance(tool, AgentResponseToQuestionTool):
                return tool
        return None

    def run_discussion_round(self) -> None:
        """Run one discussion round:
        - Ask all agents if they want to speak (and optionally direct a question to someone)
        - Randomize the order of agents who said yes
        - For each, create an AskAgentIfWantsToSpeakEventPublic
          - If a directed question is present, prompt that target and log an AgentResponseToQuestioningEventPublic
        """
        agent_ids = list(self.agents_by_id.keys())

        # 1) Ask intent to speak
        speak_intents: list[tuple[str, AskAgentIfWantsToSpeakTool]] = []
        for agent_id in agent_ids:
            agent = self.agents_by_id[agent_id]
            others = [aid for aid in agent_ids if aid != agent_id]
            intent = self._prompt_agent_for_speak_intent(agent, others)
            if intent is not None:
                speak_intents.append((agent_id, intent))

        if not speak_intents:
            return

        # 2) Randomize order
        random.shuffle(speak_intents)

        # 3) Execute in order
        for speaker_id, intent_tool in speak_intents:
            # Log the speaker's public event (question/statement and optional target)
            self.public_events.append(
                AskAgentIfWantsToSpeakEventPublic(
                    agent_id=speaker_id,
                    question_or_statement=intent_tool.question_or_statement,
                    ask_directed_question_to_agent_id=intent_tool.ask_directed_question_to_agent_id,
                )
            )

            # If there's a directed question, prompt the target for a response
            target_id = intent_tool.ask_directed_question_to_agent_id
            if target_id and target_id in self.agents_by_id:
                target_agent = self.agents_by_id[target_id]
                response_tool = self._prompt_agent_for_direct_answer(
                    target=target_agent,
                    asked_by_id=speaker_id,
                    question_text=intent_tool.question_or_statement,
                )
                if response_tool is not None:
                    self.public_events.append(
                        AgentResponseToQuestioningEventPublic(
                            agent_id=target_id,
                            in_response_to_agent_id=speaker_id,
                            response=response_tool.response,
                        )
                    )