import json
from ...models import ToolCall


class OpenAIToolCallConverter:
    def to_dict(self, data: ToolCall) -> dict:
        return {
            "id": data.tool_call_id,
            "type": "function",
            "function": {
                "name": data.tool_name,
                "arguments": json.dumps(data.input),
            },
        }

    def from_dict(self, data: dict) -> ToolCall:
        arguments = data["function"]["arguments"]
        return ToolCall(
            tool_call_id=data["id"],
            tool_name=data["function"]["name"],
            input=json.loads(arguments) if arguments else {},
        )
