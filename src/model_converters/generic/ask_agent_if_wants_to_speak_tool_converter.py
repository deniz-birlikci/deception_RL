from ...models import AskAgentIfWantsToSpeakTool


class AskAgentIfWantsToSpeakToolConverter:
    def to_dict(self, data: AskAgentIfWantsToSpeakTool) -> dict:
        return {
            "question_or_statement": data.question_or_statement,
            "ask_directed_question_to_agent_id": data.ask_directed_question_to_agent_id,
        }

    def from_dict(self, data: dict) -> AskAgentIfWantsToSpeakTool:
        return AskAgentIfWantsToSpeakTool(
            tool_type="ask-agent-if-wants-to-speak",
            question_or_statement=data.get("question_or_statement"),
            ask_directed_question_to_agent_id=data.get(
                "ask_directed_question_to_agent_id"
            ),
        )
