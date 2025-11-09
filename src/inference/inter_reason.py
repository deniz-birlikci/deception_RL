"""
Qwen2.5-7B reasoning + tool call where reasoning precedes the tool call.

The model is prompted to explain its thought process in plain text first, then issue
an actual tool call (OpenAI function call) that we capture separately. The final
result keeps reasoning and tool call distinct.
"""

import modal
import time

app = modal.App("qwen7b-intermediate-reasoning")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("procps")
    .pip_install(
        "openpipe-art[backend]",
        "openai>=1.65.5",
    )
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MAKE_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "make_move",
        "description": "Return your reasoning and chosen move for the 2048 board.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Short summary explaining why this move is best.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down"],
                    "description": "Direction to move the tiles.",
                },
            },
            "required": ["reasoning", "direction"],
            "additionalProperties": False,
        },
    },
}


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    secrets=[modal.Secret.from_name("art-secrets")],
)
async def inference_with_reasoning(board_state: str):
    """
    Run Qwen7B inference to obtain reasoning first, then a tool call.
    """
    import art
    from art.local import LocalBackend

    start_time = time.perf_counter()

    model = art.TrainableModel(
        name="qwen7b-inter-reason",
        project="inference-test",
        base_model=MODEL_NAME,
    )

    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(max_seq_length=8192),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
            additional_config={
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
            },
        ),
    )

    backend = LocalBackend()
    await model.register(backend)
    await backend._get_service(model)  # type: ignore[attr-defined]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert 2048 strategist. Examine the board carefully. "
                "First explain your reasoning in natural language. Only after you finish "
                "your reasoning should you call the make_move function exactly once. "
                "When calling the tool, include both a 'reasoning' field and the selected 'direction'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Board:\n"
                f"{board_state}\n\n"
                "Explain your reasoning, then call the tool."
            ),
        },
    ]

    client = model.openai_client()

    params = {
        "model": model.name,
        "messages": messages,
        "tools": [MAKE_MOVE_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "make_move"}}, # force tool
        "max_completion_tokens": 512,
        "temperature": 0.7,
    }

    response = await client.chat.completions.create(**params)

    choice = response.choices[0]
    message = choice.message

    reasoning_text = (message.content or "").strip() if hasattr(message, "content") else ""

    tool_call = None
    if message.tool_calls:
        tc = message.tool_calls[0]
        tool_call = {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }

    return {
        "reasoning": reasoning_text,
        "tool_call": tool_call,
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if hasattr(response, "usage") else None,
        "generation_time_seconds": time.perf_counter() - start_time,
    }


@app.local_entrypoint()
def main():
    """
    Run inference and display reasoning + tool call.
    """
    board_state = """
 64| 32| 16|  8
 32| 64| 32| 16
 16| 32| 64| 32
  8| 16| 32|128
"""

    print("=" * 80)
    print("QWEN7B REASONING + TOOL CALL")
    print("=" * 80)
    print("\nBoard State:")
    print(board_state)
    print("=" * 80)

    result = inference_with_reasoning.remote(board_state)

    print("\n" + "=" * 80)
    print("REASONING:")
    print("=" * 80)
    print(result["reasoning"])

    print("\n" + "=" * 80)
    print("TOOL CALL:")
    print("=" * 80)
    if result["tool_call"]:
        print(result["tool_call"])
    else:
        print("No tool call generated")

    print("\n" + "=" * 80)
    print("METRICS:")
    print("=" * 80)
    print(f"Generation time: {result['generation_time_seconds']:.2f}s")
    print(f"Finish reason: {result['finish_reason']}")
    if result.get("usage"):
        print(f"Tokens: {result['usage']}")
    print("=" * 80)
