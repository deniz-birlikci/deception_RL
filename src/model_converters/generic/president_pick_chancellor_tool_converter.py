from ...models import PresidentPickChancellorTool


class PresidentPickChancellorToolConverter:

    def to_dict(self, data: PresidentPickChancellorTool) -> dict:
        return {"agent_id": data.agent_id}

    def from_dict(self, data: dict) -> PresidentPickChancellorTool:
        return PresidentPickChancellorTool(
            tool_type="president-pick-chancellor", agent_id=data["agent_id"]
        )
