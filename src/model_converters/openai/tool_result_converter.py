from ...models import ToolResult


class OpenAIToolResultConverter:

    def to_dict(self, data: ToolResult) -> dict:
        return {
            "role": "tool",
            "tool_call_id": data.tool_call_id,
            "content": data.output,
        }

    def from_dict(self, data: dict) -> ToolResult:
        raise NotImplementedError(
            f"openai format does not support deserializing ToolResult from dict: {data}"
        )
