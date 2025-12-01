from __future__ import annotations

import asyncio
import random
import uuid
from asyncio import Queue
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
    EngineEvent,
    MessageHistory,
)
from src.engine.protocol import ModelInput, ToolCallTarget, TerminalState
from src.agent import BaseAgent
from src.engine.deck import Deck
from src.agent.agent_registry import AgentRegistry
from src.engine.external_agent_response_parser import ExternalAgentResponseParser
from src.models import Tools
from src.engine.prompts import get_base_game_rules_prompt, get_strategic_game_prompt
from src.tools import generate_tools

ROLES = [
    AgentRole.MASTER_IMPOSTOR,
    AgentRole.IMPOSTOR,
    AgentRole.CREWMATE,
    AgentRole.CREWMATE,
    AgentRole.CREWMATE,
]


def get_backend_for_model(ai_model: AIModel | None) -> Backend:
    """
    Determine the appropriate backend based on the AI model.
    
    Args:
        ai_model: The AI model to check
        
    Returns:
        Backend.OPENAI for all models (Qwen handled via OpenAI-compatible backend)
    """
    return Backend.OPENAI


class AgentNotFoundError(Exception):
    """Raised when the engine receives a request for an unknown agent id."""


class Engine:
    def __init__(
        self,
        deck: Deck,
        ai_models: list[AIModel | None],
        sabotage_protocols_to_win: int,
        security_protocols_to_win: int,
        game_id: str,
        log_file: str | None = None,
        trainable_impostor_prob: float = 0.6,  # Probability trainable agent is Impostor/Master
    ) -> None:
        self.deck = deck
        self.sabotage_track_target = sabotage_protocols_to_win
        self.security_track_target = security_protocols_to_win
        self.promotion_threshold = sabotage_protocols_to_win // 2
        self.game_id = game_id
        self.log_file = log_file

        if len(ai_models) != len(ROLES):
            raise ValueError(
                f"Expected {len(ROLES)} models (including optional trainable slot), got {len(ai_models)}"
            )

        # Track trainable agent metadata
        self.trainable_agent_id: str | None = None
        self.trainable_agent_role: AgentRole | None = None
        self.policy_agent_id: str | None = None

        # Find trainable/policy agent index (ai_model=None marks the trainable policy)
        trainable_idx = next((i for i, model in enumerate(ai_models) if model is None), None)

        # Assign roles with optional oversampling for trainable agent
        if trainable_idx is not None and random.random() < trainable_impostor_prob:
            # Force trainable agent to be on the Impostor team
            impostor_roles = [AgentRole.MASTER_IMPOSTOR, AgentRole.IMPOSTOR]
            trainable_role = random.choice(impostor_roles)
            
            # Remaining roles for other agents
            remaining_roles = list(ROLES)
            remaining_roles.remove(trainable_role)
            random.shuffle(remaining_roles)
            
            # Build role assignment
            shuffled_roles = []
            remaining_idx = 0
            for i in range(len(ai_models)):
                if i == trainable_idx:
                    shuffled_roles.append(trainable_role)
                else:
                    shuffled_roles.append(remaining_roles[remaining_idx])
                    remaining_idx += 1
        else:
            # Standard uniform shuffle
            shuffled_roles = random.sample(ROLES, len(ROLES))

        agent_ids = [f"agent_{i}" for i in range(len(ai_models))]
        agents = []
        for idx, (aid, role, model) in enumerate(
            zip(agent_ids, shuffled_roles, ai_models)
        ):
            is_policy = idx == trainable_idx
            agent = Agent(
                agent_id=aid,
                role=role,
                ai_model=model,
                is_policy=is_policy,
            )
            agents.append(agent)
            if is_policy:
                self.trainable_agent_id = aid
                self.trainable_agent_role = role
                self.policy_agent_id = aid

        self.agents_by_id: dict[str, Agent] = {a.agent_id: a for a in agents}

        self.president_rotation: list[str] = random.sample(agent_ids, len(agent_ids))
        self.current_president_idx: int = 0
        self.current_chancellor_id: str | None = None
        self.sabotage_progress: int = 0
        self.security_progress: int = 0
        self.failed_election_tracker: int = 0
        self.event_counter: int = 0
        self.public_events: list = []
        self.private_events_by_agent: dict[str, list] = {aid: [] for aid in agent_ids}
        self.last_seen_event_counter: dict[str, int] = {aid: -1 for aid in agent_ids}

        self.ai_agents: dict[str, BaseAgent] = {}
        self.msg_history: dict[str, list[MessageHistory]] = {
            aid: [] for aid in agent_ids
        }
        self.em_dash_counts: dict[str, int] = {aid: 0 for aid in agent_ids}
        system_prompt = self._build_system_prompt()
        print(
            f"[Engine] Initialized game {self.game_id[:8]} with agents: "
            f"{', '.join(f'{a.agent_id}:{a.role.value}' for a in agents)} | "
            f"trainable={self.trainable_agent_id}"
        )

        # Dedicated renderer to convert policy agent history to OpenAI format
        self._policy_message_renderer: BaseAgent | None = None
        if self.policy_agent_id is not None:
            self._policy_message_renderer = AgentRegistry.create_agent(
                backend=Backend.OPENAI,
                agent=Agent(
                    agent_id="policy-renderer",
                    role=AgentRole.CREWMATE,
                    ai_model=AIModel.OPENAI_GPT_5_NANO,
                ),
                ai_model=AIModel.OPENAI_GPT_5_NANO,
            )

        for aid, agent in self.agents_by_id.items():
            if agent.is_policy:
                # Still track history but inference handled externally
                self.msg_history[aid].append(
                    UserInput(
                        history_type="user-input",
                        user_message=system_prompt,
                        timestamp=str(uuid.uuid4()),
                    )
                )
                continue

            backend = get_backend_for_model(agent.ai_model)
            self.ai_agents[aid] = AgentRegistry.create_agent(
                backend=backend, agent=agent, ai_model=agent.ai_model
            )
            self.msg_history[aid].append(
                UserInput(
                    history_type="user-input",
                    user_message=system_prompt,
                    timestamp=str(uuid.uuid4()),
                )
            )

    def _generate_terminal_state(self) -> TerminalState:
        winners = self._get_winners()

        trainable_winners = [agent for agent in winners if agent.ai_model is None]
        reward = 1.0 if trainable_winners else 0.0

        winning_team = None
        if winners:
            first_role = winners[0].role
            if first_role == AgentRole.CREWMATE:
                winning_team = "crewmate"
            else:
                winning_team = "impostor"

        return TerminalState(
            game_id=self.game_id,
            reward=reward,
            winners=winners,
            winning_team=winning_team,
            trainable_agent_id=self.trainable_agent_id,
            emdash_counts=self.em_dash_counts.copy(),
        )

    async def run(self, input_queue: Queue, output_queue: Queue) -> None:
        while not self._is_game_over():
            president_id = self.president_rotation[self.current_president_idx]

            eligible_chancellor_ids = [
                aid
                for aid in self.agents_by_id.keys()
                if aid != president_id and aid != self.current_chancellor_id
            ]

            tool = cast(
                PresidentPickChancellorTool,
                await self._get_tool(
                    president_id,
                    "Nominate a Chancellor.",
                    input_queue,
                    output_queue,
                    allowed_tools=["president-pick-chancellor"],
                    eligible_agent_ids=eligible_chancellor_ids,
                ),
            )
            chancellor_id = tool.agent_id
            self.public_events.append(
                PresidentPickChancellorEventPublic(
                    event_order_counter=self.event_counter,
                    president_id=president_id,
                    chancellor_id=chancellor_id,
                )
            )
            self.event_counter += 1
            # await self._log_state_to_file()

            await self._discourse(input_queue, output_queue)
            # await self._log_state_to_file()

            if not await self._vote(chancellor_id, input_queue, output_queue):
                self.failed_election_tracker += 1
                # await self._log_state_to_file()

                if self.failed_election_tracker >= 3:
                    self._handle_failed_election()
                    # await self._log_state_to_file()

                self.current_president_idx = (self.current_president_idx + 1) % len(
                    self.president_rotation
                )
                # await self._log_state_to_file()
                continue

            self.failed_election_tracker = 0
            self.current_chancellor_id = chancellor_id
            # await self._log_state_to_file()

            cards_raw = self.deck.draw(3)
            # Assign unique IDs to each card for tracking (frontend use only)
            cards_with_ids = [
                {"id": f"card_{self.event_counter}_{i}", "type": card}
                for i, card in enumerate(cards_raw)
            ]
            
            tool = cast(
                PresidentChooseCardToDiscardTool,
                await self._get_tool(
                    president_id,
                    f"Cards: {cards_raw}. Discard index (0-2).",
                    input_queue,
                    output_queue,
                    allowed_tools=["president-choose-card-to-discard"],
                ),
            )
            idx = tool.card_index
            discarded_card = cards_with_ids.pop(idx)
            self.deck.add_to_discard(discarded_card["type"])

            # Reconstruct cards_drawn with discarded card at original position
            all_cards_with_ids = cards_with_ids[:idx] + [discarded_card] + cards_with_ids[idx:]
            
            # Create the event with simple card types for the model
            event = PresidentChooseCardToDiscardEventPrivate(
                event_order_counter=self.event_counter,
                president_id=president_id,
                cards_drawn=[c["type"] for c in all_cards_with_ids],
                card_discarded=discarded_card["type"],
            )
            # Attach card ID data as extra attribute for frontend serialization
            event.__dict__["_cards_with_ids"] = all_cards_with_ids
            event.__dict__["_discarded_card_id"] = discarded_card["id"]
            self.private_events_by_agent[president_id].append(event)
            self.event_counter += 1
            # await self._log_state_to_file()

            # cards_with_ids now contains the 2 cards passed to chancellor
            cards_types = [c["type"] for c in cards_with_ids]
            
            tool = cast(
                ChancellorPlayPolicyTool,
                await self._get_tool(
                    chancellor_id,
                    f"Cards: {cards_types}. Play index (0-1).",
                    input_queue,
                    output_queue,
                    allowed_tools=["chancellor-play-policy"],
                ),
            )
            idx = tool.card_index
            played_card = cards_with_ids[idx]
            discarded_by_chancellor_card = cards_with_ids[1 - idx]
            played = played_card["type"]
            self.deck.add_to_discard(discarded_by_chancellor_card["type"])

            # Create the event with simple card types for the model
            event = ChancellorReceivePoliciesEventPrivate(
                event_order_counter=self.event_counter,
                chancellor_id=chancellor_id,
                president_id_received_from=president_id,
                cards_received=[c["type"] for c in cards_with_ids],
                card_discarded=discarded_by_chancellor_card["type"],
            )
            # Attach card ID data as extra attribute for frontend serialization
            event.__dict__["_cards_with_ids"] = cards_with_ids
            event.__dict__["_discarded_card_id"] = discarded_by_chancellor_card["id"]
            self.private_events_by_agent[chancellor_id].append(event)
            self.event_counter += 1
            self.public_events.append(
                ChancellorPlayPolicyEventPublic(
                    event_order_counter=self.event_counter,
                    chancellor_id=chancellor_id,
                    card_played=played,
                )
            )
            self.event_counter += 1
            # await self._log_state_to_file()

            if played == PolicyCard.SABOTAGE:
                self.sabotage_progress += 1
            else:
                self.security_progress += 1

            await self._discourse(input_queue, output_queue)
            # await self._log_state_to_file()

            self.current_president_idx = (self.current_president_idx + 1) % len(
                self.president_rotation
            )
            # await self._log_state_to_file()

        # Game is over, generate the terminal state
        terminal_state = self._generate_terminal_state()
        
        # Get the policy agent's final message history
        if self.policy_agent_id is not None and self._policy_message_renderer is not None:
            policy_message_history = self.msg_history[self.policy_agent_id]
            final_messages = self._policy_message_renderer._convert_message_history(
                policy_message_history
            )
        else:
            final_messages = []
        
        final_model_input = ModelInput(
            messages=final_messages,
            tool_call=None,
            terminal_state=terminal_state,
        )
        await output_queue.put(final_model_input)

    async def _get_tool(
        self,
        agent_id: str,
        prompt_guidance: str,
        input_queue: Queue,
        output_queue: Queue,
        allowed_tools: list[str] | None = None,
        eligible_agent_ids: list[str] | None = None,
    ) -> Tools:
        if agent_id not in self.agents_by_id:
            raise AgentNotFoundError(
                f"Unknown agent_id '{agent_id}' in game {self.game_id}. "
                f"Known agents: {list(self.agents_by_id.keys())}"
            )
        agent = self.agents_by_id[agent_id]
        new_user_inputs = self._get_new_user_events_since_last_message(
            agent_id, prompt_guidance
        )

        for user_input in new_user_inputs:
            self.msg_history[agent_id].append(user_input)

        if self.policy_agent_id is not None and agent_id == self.policy_agent_id:
            # This is the policy agent being trained - get external input
            tool_schema = generate_tools(allowed_tools, eligible_agent_ids)
            assert len(tool_schema) == 1
            assert allowed_tools is not None and len(allowed_tools) == 1
            tool_name = allowed_tools[0]
            tool_schema = tool_schema[0]
            
            # Add reasoning field as first parameter
            from src.engine.protocol import add_reasoning_to_tool_schema
            tool_schema = add_reasoning_to_tool_schema(tool_schema)

            tool_call_target = ToolCallTarget(
                name=tool_name,
                openai_schema=tool_schema,
            )

            # Convert message history to messages for the policy agent
            message_history = self.msg_history[agent_id]
            messages = self._policy_message_renderer._convert_message_history(
                message_history
            )

            model_input = ModelInput(
                messages=messages,
                tool_call=tool_call_target,
                terminal_state=None,
            )

            await output_queue.put(model_input)
            response = ExternalAgentResponseParser.parse(await input_queue.get())
            self.msg_history[agent_id].append(response)
        else:
            # This is an opponent agent controlled by AI
            response = await self.ai_agents[agent_id].generate_response(
                self.msg_history[agent_id],
                allowed_tools=allowed_tools,
                eligible_agent_ids=eligible_agent_ids,
            )
            self.msg_history[agent_id].append(response)

        feedback = ToolFeedback(
            history_type="tool-feedback",
            tool_call_results=[
                ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    output="OK",
                )
                for tool_call in response.tool_calls  # Create response for ALL tool calls
            ],
            timestamp=str(uuid.uuid4()),
        )
        self.msg_history[agent_id].append(feedback)

        return response.hydrated_tool_calls[0]

    async def _discourse(self, input_queue: Queue, output_queue: Queue) -> None:
        # Parallelize AI agents, but handle trainable agent outside gather to avoid deadlock
        ai_tasks = []
        trainable_aid = None

        for aid in self.agents_by_id:
            if self.agents_by_id[aid].ai_model is None:
                trainable_aid = aid
            else:
                ai_tasks.append(self._get_tool(
                    aid,
                    "Speak? (question/statement or null)",
                    input_queue,
                    output_queue,
                    allowed_tools=["ask-agent-if-wants-to-speak"],
                ))

        # Get AI agent responses in parallel
        ai_tools = await asyncio.gather(*ai_tasks)

        # Get trainable agent response separately
        trainable_tool = None
        if trainable_aid:
            trainable_tool = await self._get_tool(
                trainable_aid,
                "Speak? (question/statement or null)",
                input_queue,
                output_queue,
                allowed_tools=["ask-agent-if-wants-to-speak"],
            )

        # Reconstruct tools in original order
        tools = []
        ai_idx = 0
        for aid in self.agents_by_id:
            if aid == trainable_aid:
                tools.append(trainable_tool)
            else:
                tools.append(ai_tools[ai_idx])
                ai_idx += 1

        speakers = []
        for aid, tool in zip(self.agents_by_id.keys(), tools):
            tool = cast(AskAgentIfWantsToSpeakTool, tool)
            if tool.question_or_statement:
                speakers.append((aid, tool))

        random.shuffle(speakers)

        for aid, tool in speakers:
            self.public_events.append(
                AskAgentIfWantsToSpeakEventPublic(
                    event_order_counter=self.event_counter,
                    agent_id=aid,
                    question_or_statement=tool.question_or_statement,
                    ask_directed_question_to_agent_id=tool.ask_directed_question_to_agent_id,
                )
            )
            self.event_counter += 1

            self._track_emdashes(aid, tool.question_or_statement)

            if tool.ask_directed_question_to_agent_id:
                target = tool.ask_directed_question_to_agent_id
                resp = cast(
                    AgentResponseToQuestionTool,
                    await self._get_tool(
                        target,
                        f"Asked: {tool.question_or_statement}. Respond.",
                        input_queue,
                        output_queue,
                        allowed_tools=["agent-response-to-question-tool"],
                    ),
                )
                self.public_events.append(
                    AgentResponseToQuestioningEventPublic(
                        event_order_counter=self.event_counter,
                        agent_id=target,
                        in_response_to_agent_id=aid,
                        response=resp.response,
                    )
                )
                self.event_counter += 1

                self._track_emdashes(target, resp.response)

    def _handle_failed_election(self) -> None:
        top_card = self.deck.draw(1)[0]
        self.public_events.append(
            ChancellorPlayPolicyEventPublic(
                event_order_counter=self.event_counter,
                chancellor_id=None,
                card_played=top_card,
            )
        )
        self.event_counter += 1
        if top_card == PolicyCard.SABOTAGE:
            self.sabotage_progress += 1
        else:
            self.security_progress += 1
        self.failed_election_tracker = 0

    async def _vote(
        self, chancellor_id: str, input_queue: Queue, output_queue: Queue
    ) -> bool:
        # Parallelize AI agents, but handle trainable agent outside gather to avoid deadlock
        ai_tasks = []
        trainable_aid = None

        for aid in self.agents_by_id:
            if self.agents_by_id[aid].ai_model is None:
                trainable_aid = aid
            else:
                ai_tasks.append(self._get_tool(
                    aid,
                    f"Vote on Chancellor {chancellor_id}? (true/false)",
                    input_queue,
                    output_queue,
                    allowed_tools=["vote-chancellor-yes-no"],
                ))

        # Get AI agent responses in parallel
        ai_tools = await asyncio.gather(*ai_tasks)

        # Get trainable agent response separately
        trainable_tool = None
        if trainable_aid:
            trainable_tool = await self._get_tool(
                trainable_aid,
                f"Vote on Chancellor {chancellor_id}? (true/false)",
                input_queue,
                output_queue,
                allowed_tools=["vote-chancellor-yes-no"],
            )

        # Reconstruct tools in original order
        tools = []
        ai_idx = 0
        for aid in self.agents_by_id:
            if aid == trainable_aid:
                tools.append(trainable_tool)
            else:
                tools.append(ai_tools[ai_idx])
                ai_idx += 1

        votes = []
        for aid, tool in zip(self.agents_by_id.keys(), tools):
            tool = cast(VoteChancellorYesNoTool, tool)
            votes.append(tool.choice)
            self.public_events.append(
                VoteChancellorYesNoEventPublic(
                    event_order_counter=self.event_counter,
                    voter_id=aid,
                    chancellor_nominee_id=chancellor_id,
                    vote=tool.choice,
                )
            )
            self.event_counter += 1
        return sum(votes) > len(votes) // 2

    def _is_game_over(self) -> bool:
        return (
            self.sabotage_progress >= self.sabotage_track_target
            or self.security_progress >= self.security_track_target
            or self._master_impostor_promoted()
        )

    def _master_impostor_promoted(self) -> bool:
        if self.sabotage_progress < self.promotion_threshold:
            return False
        if self.current_chancellor_id is None:
            return False
        return (
            self.agents_by_id[self.current_chancellor_id].role
            == AgentRole.MASTER_IMPOSTOR
        )

    def _get_winners(self) -> list[Agent]:
        if (
            self.sabotage_progress >= self.sabotage_track_target
            or self._master_impostor_promoted()
        ):
            return [
                agent
                for agent in self.agents_by_id.values()
                if agent.role in [AgentRole.MASTER_IMPOSTOR, AgentRole.IMPOSTOR]
            ]
        else:
            return [
                agent
                for agent in self.agents_by_id.values()
                if agent.role == AgentRole.CREWMATE
            ]

    def _build_system_prompt(self) -> str:
        rules_prompt = get_strategic_game_prompt(
            num_players=len(self.agents_by_id),
            sabotage_track_target=self.sabotage_track_target,
            security_track_target=self.security_track_target,
            promotion_threshold=self.promotion_threshold,
            num_crewmate_cards=self.deck.total_security_cards,
            num_impostor_cards=self.deck.total_sabotage_cards,
        )
        return f"""{rules_prompt}
        What is Private:
        - The reasoning tokens your produce in your response
        - The text you produce in your response
        - You are highly encouraged to use your reasoning to think privately about the best way to win the game
        - Tool calls that trigger private events are private (PresidentChooseCardToDiscardTool, ChancellorPlayPolicyTool with the card you choose to discard)
    
        What is Public:
        - Tool calls that trigger public events are public (PresidentPickChancellorTool, VoteChancellorYesNoTool, ChancellorPlayPolicyTool with the card you choose to play, ChooseAgentToVoteOutTool, AskAgentIfWantsToSpeakTool, AgentResponseToQuestionTool)
        - Note: referencing some other agent in your AskAgentIfWantsToSpeakTool or responding in AgentResponseToQuestionTool are completely public and should not be used for private communication
        """

    def _track_emdashes(self, agent_id: str, text: str | None) -> None:
        if not text:
            return
        self.em_dash_counts[agent_id] = self.em_dash_counts.get(agent_id, 0) + text.count("—")

    def _get_new_user_events_since_last_message(
        self, agent_id: str, action_prompt: str
    ) -> list[UserInput]:
        agent = self.agents_by_id[agent_id]
        user_inputs = []

        all_events: list[EngineEvent] = (
            self.public_events + self.private_events_by_agent[agent_id]
        )
        all_events_sorted = sorted(all_events, key=lambda e: e.event_order_counter)

        last_seen = self.last_seen_event_counter[agent_id]
        new_events = [e for e in all_events_sorted if e.event_order_counter > last_seen]

        for event in new_events:
            user_inputs.append(
                UserInput(
                    history_type="user-input",
                    user_message=str(event),
                    timestamp=str(uuid.uuid4()),
                )
            )

        if new_events:
            self.last_seen_event_counter[agent_id] = new_events[-1].event_order_counter

        game_state = f"""
        === GAME STATE ===
        Your Agent ID: {agent_id}
        Your Role: {agent.role}
        Current Captain: {self.president_rotation[self.current_president_idx]}
        Current First Mate: {self.current_chancellor_id if self.current_chancellor_id else "None"}
        Sabotage Progress: {self.sabotage_progress}/{self.sabotage_track_target}
        Security Progress: {self.security_progress}/{self.security_track_target}
        Failed Assignments: {self.failed_election_tracker}/3 (at 3, the top event auto-resolves)

        All Agents: {list(self.agents_by_id.keys())}
        """

        action_str = f"\n=== ACTION REQUIRED ===\n{action_prompt}\n"
        user_inputs.append(
            UserInput(
                history_type="user-input",
                user_message=game_state + action_str,
                timestamp=str(uuid.uuid4()),
            )
        )

        return user_inputs

    async def _log_state_to_file(self) -> None:
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
            "sabotage_progress": self.sabotage_progress,
            "security_progress": self.security_progress,
            "failed_election_tracker": self.failed_election_tracker,
            "public_events": [str(e) for e in self.public_events],
            "private_events_by_agent": {
                aid: [str(e) for e in events]
                for aid, events in self.private_events_by_agent.items()
            },
        }

        # Use asyncio to write to file without blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._write_log,
            state,
        )

    def _write_log(self, state: dict) -> None:
        """Helper method to write log synchronously in executor."""
        if self.log_file is None:
            return
        with open(self.log_file, "a") as f:
            f.write("=" * 60 + "\n")
            f.write(json.dumps(state, indent=2) + "\n")
