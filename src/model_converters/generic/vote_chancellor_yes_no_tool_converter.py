from ...models import VoteChancellorYesNoTool


class VoteChancellorYesNoToolConverter:

    def to_dict(self, data: VoteChancellorYesNoTool) -> dict:
        return {"choice": data.choice}

    def from_dict(self, data: dict) -> VoteChancellorYesNoTool:
        return VoteChancellorYesNoTool(
            tool_type="vote-chancellor-yes-no", choice=data["choice"]
        )
