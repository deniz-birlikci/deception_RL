from __future__ import annotations

from src.engine.engine_api import EngineAPI
from src.engine.deck import Deck
from src.models import AIModel
from src.engine.protocol import ModelOutput
import uuid
import asyncio


async def main():
    api = EngineAPI()

    deck = Deck()

    ai_models: list[AIModel | None] = [
        AIModel.OPENAI_GPT_5_MINI,
        AIModel.OPENAI_GPT_5_MINI,
        AIModel.OPENAI_GPT_5_MINI,
        AIModel.OPENAI_GPT_5_MINI,
        None,  # Treat last slot as the human/policy player
    ]

    game_id = str(uuid.uuid4())
    result = await api.create(
        game_id=game_id,
        deck=deck,
        ai_models=ai_models,
        log_file="game_log.txt"
    )

    print("=== GAME STARTED ===")
    print(result)
    print()

    while isinstance(result, str):
        user_input = input("Enter your response (JSON): ")
        game_id = list(api.games.keys())[0]
        model_output = ModelOutput(function_calling_json=user_input)
        result = await api.execute(game_id, model_output)
        print()
        print("=== RESPONSE ===")
        print(result)
        print()

    print("=== GAME OVER ===")
    print(f"Winners: {result}")


if __name__ == "__main__":
    asyncio.run(main())
