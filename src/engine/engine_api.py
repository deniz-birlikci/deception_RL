import asyncio
import random
import traceback
from asyncio import Queue
from typing import Dict
from src.models import AIModel, Agent, AgentRole
from src.engine.engine import Engine, ROLES
from src.engine.deck import Deck
from src.engine.protocol import ModelInput, ModelOutput, TerminalState

DEFAULT_OPPONENT_MODELS: list[AIModel] = [
    AIModel.OPENAI_GPT_5,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_NANO,
    AIModel.OPENAI_GPT_4_1,
]


class EngineAPI:
    def __init__(self):
        self.games: Dict[str, tuple[Queue, Queue]] = {}
        self.engines: Dict[str, Engine] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    async def create(
        self,
        game_id: str,
        deck: Deck,
        opponent_models: list[AIModel] = DEFAULT_OPPONENT_MODELS,
        policy_model: AIModel | None = None,
        fascist_policies_to_win: int = 4,
        liberal_policies_to_win: int = 3,
        log_file: str | None = None,
    ) -> ModelInput:
        input_queue: Queue = Queue()
        output_queue: Queue = Queue()

        self.games[game_id] = (input_queue, output_queue)
        
        # Create agents with random role assignment
        shuffled_roles = random.sample(ROLES, len(ROLES))
        agent_ids = [f"agent_{i}" for i in range(len(ROLES))]
        
        # All opponents + policy agent
        all_models = opponent_models + ([policy_model] if policy_model else [])
        assert len(all_models) == len(ROLES)
        
        agents = [
            Agent(
                agent_id=aid,
                role=role,
                ai_model=model,
                is_policy=(policy_model is not None and i == len(opponent_models))  # Last agent is policy
            )
            for i, (aid, role, model) in enumerate(zip(agent_ids, shuffled_roles, all_models))
        ]

        engine = Engine(
            deck=deck,
            agents=agents,
            fascist_policies_to_win=fascist_policies_to_win,
            liberal_policies_to_win=liberal_policies_to_win,
            game_id=game_id,
            log_file=log_file,
        )
        self.engines[game_id] = engine

        # Create asyncio task instead of thread
        print(f"Creating engine task for game {game_id}")
        task = asyncio.create_task(
            self._run_engine(game_id, engine, input_queue, output_queue)
        )
        self.tasks[game_id] = task

        # Await the initial message from the game engine
        model_input = await output_queue.get()
        assert isinstance(model_input, ModelInput)
        return model_input

    async def execute(self, game_id: str, model_output: ModelOutput) -> ModelInput:
        # First check that the game exists
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")

        # Obtain the input queue and output queue for the game
        input_queue, output_queue = self.games[game_id]

        # Then, place the model output on the input queue
        await input_queue.put(model_output)

        # Await the next message from the game engine
        model_input = await output_queue.get()
        assert isinstance(model_input, ModelInput)
        return model_input

    async def _run_engine(
        self,
        game_id: str,
        engine: Engine,
        input_queue: Queue,
        output_queue: Queue,
    ) -> None:
        print(f"Running engine task for game {game_id}")
        try:
            await engine.run(input_queue, output_queue)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Engine error: {str(e)}\n\nStack trace:\n{tb}")
            await output_queue.put(f"Engine error: {str(e)}\n\nStack trace:\n{tb}")
        finally:
            if game_id in self.games:
                del self.games[game_id]
            if game_id in self.engines:
                del self.engines[game_id]
            if game_id in self.tasks:
                del self.tasks[game_id]

    def get_game_ids(self) -> list[str]:
        return list(self.games.keys())

    def game_exists(self, game_id: str) -> bool:
        return game_id in self.games

    def finalize(self, game_id: str, terminal_state: TerminalState) -> None:
        pass
