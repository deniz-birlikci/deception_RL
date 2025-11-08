from ...models import AgentResponseToQuestionTool


class AgentResponseToQuestionToolConverter:
    def to_dict(self, data: AgentResponseToQuestionTool) -> dict:
        return {
            "response": data.response,
        }

    def from_dict(self, data: dict) -> AgentResponseToQuestionTool:
        return AgentResponseToQuestionTool(
            tool_type="agent-response-to-question-tool",
            response=data["response"],
        )
