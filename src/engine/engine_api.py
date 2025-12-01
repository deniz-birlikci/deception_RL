from __future__ import annotations

import asyncio
import traceback
from asyncio import Queue
from typing import Dict
from src.models import AIModel, AgentRole
from src.engine.engine import Engine, AgentNotFoundError
from src.engine.deck import Deck
from src.engine.protocol import ModelInput, ModelOutput, TerminalState

DEFAULT_AI_MODELS = [
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_MINI,
    None,
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
        ai_models: list[AIModel | None] = DEFAULT_AI_MODELS,
        sabotage_protocols_to_win: int = 4,
        security_protocols_to_win: int = 3,
        log_file: str | None = None,
        trainable_impostor_prob: float = 0.6,
    ) -> ModelInput:
        input_queue: Queue = Queue()
        output_queue: Queue = Queue()

        self.games[game_id] = (input_queue, output_queue)

        engine = Engine(
            deck=deck,
            ai_models=ai_models,
            sabotage_protocols_to_win=sabotage_protocols_to_win,
            security_protocols_to_win=security_protocols_to_win,
            game_id=game_id,
            log_file=log_file,
            trainable_impostor_prob=trainable_impostor_prob,
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
        except AgentNotFoundError as e:
            terminal = TerminalState(
                game_id=game_id,
                reward=-1.0,
                winners=[],
                winning_team=None,
            )
            await output_queue.put(
                ModelInput(messages=[], tool_call=None, terminal_state=terminal)
            )
            print(f"Agent error: {e}")
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

    def get_trainable_agent_role(self, game_id: str) -> AgentRole | None:
        engine = self.engines.get(game_id)
        if engine is None:
            return None
        return getattr(engine, "trainable_agent_role", None)

    def get_trainable_agent_id(self, game_id: str) -> str | None:
        engine = self.engines.get(game_id)
        if engine is None:
            return None
        return getattr(engine, "trainable_agent_id", None)

    def finalize(self, game_id: str, terminal_state: TerminalState) -> None:
        pass
