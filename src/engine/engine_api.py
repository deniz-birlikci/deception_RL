import uuid
import threading
from queue import Queue
from typing import Dict
from src.models import Agent, AIModel
from src.engine.engine import Engine
from src.engine.deck import Deck

DEFAULT_AI_MODELS = [
    AIModel.OPENAI_GPT_5,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_NANO,
    AIModel.OPENAI_GPT_4_1,
    None,
]


class EngineAPI:
    def __init__(self):
        self.games: Dict[str, tuple[Queue, Queue]] = {}
        self.engines: Dict[str, Engine] = {}
        self.threads: Dict[str, threading.Thread] = {}

    def create(
        self,
        deck: Deck,
        ai_models: list[AIModel | None] = DEFAULT_AI_MODELS,
        fascist_policies_to_win: int = 6,
        liberal_policies_to_win: int = 5,
    ) -> str | Agent:
        game_id = str(uuid.uuid4())

        input_queue = Queue()
        output_queue = Queue()

        self.games[game_id] = (input_queue, output_queue)

        engine = Engine(
            deck=deck,
            ai_models=ai_models,
            fascist_policies_to_win=fascist_policies_to_win,
            liberal_policies_to_win=liberal_policies_to_win,
        )
        self.engines[game_id] = engine

        thread = threading.Thread(
            target=self._run_engine,
            args=(game_id, engine, input_queue, output_queue),
            daemon=True,
        )
        self.threads[game_id] = thread
        thread.start()

        return output_queue.get()

    def execute(self, game_id: str, response: str | None) -> str | Agent:
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")

        input_queue, output_queue = self.games[game_id]

        input_queue.put(response)

        return output_queue.get()

    def _run_engine(
        self,
        game_id: str,
        engine: Engine,
        input_queue: Queue,
        output_queue: Queue,
    ) -> None:
        try:
            engine.run(input_queue, output_queue)
        except Exception as e:
            output_queue.put(f"Engine error: {str(e)}")
        finally:
            if game_id in self.games:
                del self.games[game_id]
            if game_id in self.engines:
                del self.engines[game_id]
            if game_id in self.threads:
                del self.threads[game_id]

    def get_game_ids(self) -> list[str]:
        return list(self.games.keys())

    def game_exists(self, game_id: str) -> bool:
        return game_id in self.games
