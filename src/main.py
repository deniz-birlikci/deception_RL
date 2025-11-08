from .agent import AgentRegistry
from .env import settings
from .models import UserInput, AIModel
import time


def main():

    agent = AgentRegistry.create_agent(
        backend=settings.ai.backend,
        ai_model=AIModel.OPENAI_GPT_5
    )

    response = agent.generate_response(
        [
            UserInput(
                history_type="user-input",
                timestamp=str(int(time.time() * 1000)),
                user_message="call the president choose card to discoard tool with card index 1",
            )
        ]
    )

    print(response)


if __name__ == "__main__":
    main()
