from src.engine.engine_api import EngineAPI
from src.engine.deck import Deck
from src.models import AIModel


def main():
    api = EngineAPI()

    deck = Deck()

    ai_models: list[AIModel | None] = [
        AIModel.OPENAI_GPT_5_NANO,
        AIModel.OPENAI_GPT_5_NANO,
        AIModel.OPENAI_GPT_5_NANO,
        AIModel.OPENAI_GPT_5_NANO,
        AIModel.OPENAI_GPT_5_NANO,
    ]

    result = api.create(deck=deck, ai_models=ai_models, log_file="game_log.txt")

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
    main()
