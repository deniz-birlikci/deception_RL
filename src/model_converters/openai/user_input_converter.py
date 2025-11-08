from ...models import UserInput, AIModel
from ...env import settings


class OpenAIUserInputConverter:

    def __init__(
        self,
        ai_model: AIModel | None = None,
    ) -> None:
        self.ai_model = ai_model or settings.ai.model_id

    def to_list_dict(self, data: UserInput) -> list[dict]:
        page_content = [
            {"type": "text", "text": data.user_message},
        ]

        return [{"role": "user", "content": page_content}]

    def from_dict(self, data: dict) -> UserInput:
        raise NotImplementedError(
            f"openai format does not support deserializing UserInput from dict: {data}"
        )
