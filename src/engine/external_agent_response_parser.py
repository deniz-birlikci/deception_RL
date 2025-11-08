import json
import uuid
from src.models import AssistantResponse, ToolCall, Tools
from src.engine.protocol import ModelOutput


class ExternalAgentResponseParser:

    @staticmethod
    def parse(model_output: ModelOutput) -> AssistantResponse:
        """
        Parse a ModelOutput into an AssistantResponse object.
        
        The ModelOutput should contain:
        - function_calling_json: JSON string with tool_name and arguments
        - reasoning: Optional reasoning/thinking from the model
        
        Expected function_calling_json format:
        {
            "tool_name": "president-pick-chancellor",
            "arguments": {"agent_id": "..."}
        }
        
        The reasoning field from ModelOutput will be included in the 
        AssistantResponse for logging/debugging purposes.
        """
        # Obtain the function calling JSON and reasoning from the model output
        function_calling_json = model_output.function_calling_json
        reasoning = model_output.reasoning

        # Try to parse the function calling JSON
        try:
            data = json.loads(function_calling_json)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {function_calling_json}")

        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})

        if not tool_name:
            raise ValueError("Response must contain 'tool_name'")

        tool_call = ToolCall(
            tool_call_id=str(uuid.uuid4()),
            tool_name=tool_name,
            input=arguments,
        )

        hydrated_tool = ExternalAgentResponseParser._hydrate_tool(tool_name, arguments)

        return AssistantResponse(
            history_type="assistant-response",
            reasoning=reasoning or None,
            text_response=None,
            tool_calls=[tool_call],
            hydrated_tool_calls=[hydrated_tool],
            timestamp=str(uuid.uuid4()),
        )

    @staticmethod
    def _hydrate_tool(tool_name: str, arguments: dict) -> Tools:
        """Convert tool name and arguments into a hydrated tool object."""
        from src.models import (
            PresidentPickChancellorTool,
            VoteChancellorYesNoTool,
            PresidentChooseCardToDiscardTool,
            ChancellorPlayPolicyTool,
            ChooseAgentToVoteOutTool,
            AskAgentIfWantsToSpeakTool,
            AgentResponseToQuestionTool,
        )

        tool_map = {
            "president-pick-chancellor": PresidentPickChancellorTool,
            "vote-chancellor-yes-no": VoteChancellorYesNoTool,
            "president-choose-card-to-discard": PresidentChooseCardToDiscardTool,
            "chancellor-play-policy": ChancellorPlayPolicyTool,
            "choose-agent-to-vote-out": ChooseAgentToVoteOutTool,
            "ask-agent-if-wants-to-speak": AskAgentIfWantsToSpeakTool,
            "agent-response-to-question-tool": AgentResponseToQuestionTool,
        }

        tool_class = tool_map.get(tool_name)
        if not tool_class:
            raise ValueError(f"Unknown tool name: {tool_name}")

        return tool_class(tool_type=tool_name, **arguments)
