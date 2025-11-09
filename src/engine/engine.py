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
from src.engine.prompts import get_base_game_rules_prompt
from src.tools import generate_tools

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
        game_id: str,
        log_file: str | None = None,
    ) -> None:
        self.deck = deck
        self.fascist_policies_to_win = fascist_policies_to_win
        self.liberal_policies_to_win = liberal_policies_to_win
        self.hitler_election_threshold = fascist_policies_to_win // 2
        self.game_id = game_id
        self.log_file = log_file

        assert len(ai_models) == len(ROLES)

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
        self.event_counter: int = 0
        self.public_events: list = []
        self.private_events_by_agent: dict[str, list] = {aid: [] for aid in agent_ids}
        self.last_seen_event_counter: dict[str, int] = {aid: -1 for aid in agent_ids}

        self.ai_agents: dict[str, BaseAgent] = {}
        self.msg_history: dict[str, list[MessageHistory]] = {
            aid: [] for aid in agent_ids
        }
        system_prompt = self._build_system_prompt()

        for aid, agent in self.agents_by_id.items():
            if not agent.ai_model:
                continue
            self.ai_agents[aid] = AgentRegistry.create_agent(
                backend=Backend.OPENAI, agent=agent, ai_model=agent.ai_model
            )
            # initialize with system prompt
            self.msg_history[aid].append(
                UserInput(
                    history_type="user-input",
                    user_message=system_prompt,
                    timestamp=str(uuid.uuid4()),
                )
            )

        # Toy OpenAI Model Factory
        self._openai_agent_for_message_rendering = AgentRegistry.create_agent(
            backend=Backend.OPENAI,
            agent=Agent(
                agent_id="dummy-agent",
                role=AgentRole.LIBERAL,
                ai_model=AIModel.OPENAI_GPT_5_NANO,
            ),
            ai_model=AIModel.OPENAI_GPT_5_NANO,
        )

    def _generate_terminal_state(self) -> TerminalState:
        # TODO: check if I've won, populate it with the correct terminal state
        winners = self._get_winners()

        # There should be one single policy agent
        winning_policies = [agent for agent in winners if agent.ai_model is None]
        assert len(winning_policies) <= 1

        return TerminalState(
            game_id=self.game_id,
            reward=1.0 if len(winning_policies) == 1 else 0.0,
            winners=winning_policies,
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
            await self._log_state_to_file()

            await self._discourse(input_queue, output_queue)
            await self._log_state_to_file()

            if not await self._vote(chancellor_id, input_queue, output_queue):
                self.failed_election_tracker += 1
                await self._log_state_to_file()

                if self.failed_election_tracker >= 3:
                    self._handle_failed_election()
                    await self._log_state_to_file()

                self.current_president_idx = (self.current_president_idx + 1) % len(
                    self.president_rotation
                )
                await self._log_state_to_file()
                continue

            self.failed_election_tracker = 0
            self.current_chancellor_id = chancellor_id
            await self._log_state_to_file()

            cards = self.deck.draw(3)
            tool = cast(
                PresidentChooseCardToDiscardTool,
                await self._get_tool(
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
                    event_order_counter=self.event_counter,
                    president_id=president_id,
                    cards_drawn=cards + [discarded],
                    card_discarded=discarded,
                )
            )
            self.event_counter += 1
            await self._log_state_to_file()

            tool = cast(
                ChancellorPlayPolicyTool,
                await self._get_tool(
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
                    event_order_counter=self.event_counter,
                    chancellor_id=chancellor_id,
                    president_id_received_from=president_id,
                    cards_received=cards,
                    card_discarded=discarded_by_chancellor,
                )
            )
            self.event_counter += 1
            self.public_events.append(
                ChancellorPlayPolicyEventPublic(
                    event_order_counter=self.event_counter,
                    chancellor_id=chancellor_id,
                    card_played=played,
                )
            )
            self.event_counter += 1
            await self._log_state_to_file()

            if played == PolicyCard.FASCIST:
                self.fascist_policies_played += 1
            else:
                self.liberal_policies_played += 1

            await self._discourse(input_queue, output_queue)
            await self._log_state_to_file()

            self.current_president_idx = (self.current_president_idx + 1) % len(
                self.president_rotation
            )
            await self._log_state_to_file()

        # TODO: implement the end logic here...
        # Game is over, generate the terminal state
        terminal_state = self._generate_terminal_state()
        final_model_input = ModelInput(
            # TODO: the messages here should be the full history of the game
            # ideally... or the model can store the history itself...
            messages=[],
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
        agent = self.agents_by_id[agent_id]
        new_user_inputs = self._get_new_user_events_since_last_message(
            agent_id, prompt_guidance
        )

        for user_input in new_user_inputs:
            self.msg_history[agent_id].append(user_input)

        if agent.ai_model is None:
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

            # Convert message history to messages
            message_history = self.msg_history[agent_id]
            messages = (
                self._openai_agent_for_message_rendering._convert_message_history(
                    message_history
                )
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
        # Parallelize asking all agents if they want to speak
        ask_tasks = [
            self._get_tool(
                aid,
                "Speak? (question/statement or null)",
                input_queue,
                output_queue,
                allowed_tools=["ask-agent-if-wants-to-speak"],
            )
            for aid in self.agents_by_id
        ]
        tools = await asyncio.gather(*ask_tasks)

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
        if top_card == PolicyCard.FASCIST:
            self.fascist_policies_played += 1
        else:
            self.liberal_policies_played += 1
        self.failed_election_tracker = 0

    async def _vote(
        self, chancellor_id: str, input_queue: Queue, output_queue: Queue
    ) -> bool:
        # Parallelize all votes
        vote_tasks = [
            self._get_tool(
                aid,
                f"Vote on Chancellor {chancellor_id}? (true/false)",
                input_queue,
                output_queue,
                allowed_tools=["vote-chancellor-yes-no"],
            )
            for aid in self.agents_by_id
        ]
        tools = await asyncio.gather(*vote_tasks)

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
            self.fascist_policies_played >= self.fascist_policies_to_win
            or self.liberal_policies_played >= self.liberal_policies_to_win
            or self._hitler_elected()
        )

    def _hitler_elected(self) -> bool:
        if self.fascist_policies_played < self.hitler_election_threshold:
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

    def _build_system_prompt(self) -> str:
        rules_prompt = get_base_game_rules_prompt(
            num_players=len(self.agents_by_id),
            fascist_policies_to_win=self.fascist_policies_to_win,
            liberal_policies_to_win=self.liberal_policies_to_win,
            hitler_election_threshold=self.hitler_election_threshold,
            num_liberal_cards=self.deck.total_liberal_cards,
            num_fascist_cards=self.deck.total_fascist_cards,
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
        Current President: {self.president_rotation[self.current_president_idx]}
        Current Chancellor: {self.current_chancellor_id if self.current_chancellor_id else "None"}
        Fascist Policies Played: {self.fascist_policies_played}/{self.fascist_policies_to_win}
        Liberal Policies Played: {self.liberal_policies_played}/{self.liberal_policies_to_win}
        Failed Elections: {self.failed_election_tracker}/3 (at 3, top policy is played automatically)

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
            "fascist_policies_played": self.fascist_policies_played,
            "liberal_policies_played": self.liberal_policies_played,
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
