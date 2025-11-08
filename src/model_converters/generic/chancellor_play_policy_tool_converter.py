from ...models import ChancellorPlayPolicyTool


class ChancellorPlayPolicyToolConverter:

    def to_dict(self, data: ChancellorPlayPolicyTool) -> dict:
        return {"card_index": data.card_index}

    def from_dict(self, data: dict) -> ChancellorPlayPolicyTool:
        return ChancellorPlayPolicyTool(
            tool_type="chancellor-play-policy", card_index=data["card_index"]
        )
