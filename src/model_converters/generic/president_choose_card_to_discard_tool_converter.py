from ...models import PresidentChooseCardToDiscardTool


class PresidentChooseCardToDiscardToolConverter:

    def to_dict(self, data: PresidentChooseCardToDiscardTool) -> dict:
        return {"card_index": data.card_index}

    def from_dict(self, data: dict) -> PresidentChooseCardToDiscardTool:
        return PresidentChooseCardToDiscardTool(
            tool_type="president-choose-card-to-discard", card_index=data["card_index"]
        )
