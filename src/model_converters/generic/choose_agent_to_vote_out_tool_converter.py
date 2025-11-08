from ...models import ChooseAgentToVoteOutTool


class ChooseAgentToVoteOutToolConverter:

    def to_dict(self, data: ChooseAgentToVoteOutTool) -> dict:
        return {"agent_id": data.agent_id}

    def from_dict(self, data: dict) -> ChooseAgentToVoteOutTool:
        return ChooseAgentToVoteOutTool(
            tool_type="choose-agent-to-vote-out", agent_id=data["agent_id"]
        )
