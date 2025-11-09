from src.engine.engine_api import EngineAPI
from src.engine.deck import Deck
from src.models import AIModel
import uuid
import asyncio


async def main():
    api = EngineAPI()

    deck = Deck()

    ai_models: list[AIModel | None] = [
        AIModel.OPENAI_GPT_5,
        AIModel.OPENAI_GPT_5,
        AIModel.OPENAI_GPT_5,
        AIModel.OPENAI_GPT_5,
        AIModel.OPENAI_GPT_5,
    ]

    game_id = str(uuid.uuid4())
    result = await api.create(
        game_id=game_id, deck=deck, ai_models=ai_models, log_file="game_log.txt"
    )

    print("=== GAME STARTED ===")
    print(result)
    print()

    while isinstance(result, str):
        user_input = input("Enter your response (JSON): ")
        game_id = list(api.games.keys())[0]
        result = api.execute(game_id, user_input)
        print()
        print("=== RESPONSE ===")
        print(result)
        print()

    print("=== GAME OVER ===")
    print(f"Winners: {result}")


if __name__ == "__main__":
    asyncio.run(main())
