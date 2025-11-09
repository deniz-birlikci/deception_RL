"""
Qwen 7B (non-thinking) two-step inference.

Step 1: Get detailed reasoning about a 2048 board
Step 2: Use that reasoning to produce a <tool_call>{"move": "direction"}</tool_call>
"""

import modal
import time

app = modal.App("qwen7b-tool-call-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("procps")
    .pip_install(
        "openpipe-art[backend]",
        "openai>=1.65.5",
    )
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    secrets=[modal.Secret.from_name("art-secrets")],
)
async def two_step_inference():
    """
    Run two-step inference:
    1. First request: Get reasoning about the board
    2. Second request: Get tool call based on that reasoning
    """
    import art
    from art.local import LocalBackend

    start_time = time.perf_counter()

    model = art.TrainableModel(
        name="qwen7b-two-step",
        project="seven-inference",
        base_model=MODEL_NAME,
    )

    # Configure vLLM engine with tool calling support
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
    print("✓ Model loaded")

    board_state = """
 64| 32| 16|  8
 32| 64| 32| 16
 16| 32| 64| 32
  8| 16| 32|128
"""

    client = model.openai_client()

    # ============================================================================
    # STEP 1: Get reasoning about the board
    # ============================================================================
    print("\n--- STEP 1: Requesting reasoning ---")
    
    reasoning_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert 2048 player. Analyze the board carefully and explain "
                "your reasoning about what moves are possible and which would be best. "
                "Consider multiple options and their consequences. Be detailed in your analysis."
            ),
        },
        {
            "role": "user",
            "content": f"Analyze this 2048 board state and explain your reasoning:\n{board_state}",
        },
    ]

    reasoning_start = time.perf_counter()
    reasoning_response = await client.chat.completions.create(
        model=model.name,
        messages=reasoning_messages,
        max_completion_tokens=1024,  # Allow detailed reasoning
        temperature=0.7,
    )
    reasoning_time = time.perf_counter() - reasoning_start

    reasoning_content = reasoning_response.choices[0].message.content or ""
    print(f"✓ Reasoning generated ({len(reasoning_content)} chars)")

    # ============================================================================
    # STEP 2: Get tool call based on reasoning
    # ============================================================================
    print("\n--- STEP 2: Requesting tool call ---")
    
    tool_call_messages = [
        {
            "role": "system",
            "content": (
                "You are a 2048 game player. Based on your previous analysis, "
                "output exactly one <tool_call>{\"move\": \"direction\"}</tool_call> "
                "where direction is one of: left, right, up, down. "
                "Only output the tool call tag, nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"Here is the board:\n{board_state}",
        },
        {
            "role": "assistant",
            "content": reasoning_content,
        },
        {
            "role": "user",
            "content": "Now make your move. Output the tool call:",
        },
    ]

    tool_call_start = time.perf_counter()
    tool_call_response = await client.chat.completions.create(
        model=model.name,
        messages=tool_call_messages,
        max_completion_tokens=128,  # Short - just the tool call
        temperature=0.3,  # Lower temp for precise tool call
    )
    tool_call_time = time.perf_counter() - tool_call_start

    tool_call_content = tool_call_response.choices[0].message.content or ""
    print(f"✓ Tool call generated ({len(tool_call_content)} chars)")

    return {
        "reasoning": reasoning_content,
        "tool_call": tool_call_content,
        "reasoning_time_seconds": reasoning_time,
        "tool_call_time_seconds": tool_call_time,
        "total_time_seconds": time.perf_counter() - start_time,
        "reasoning_finish_reason": reasoning_response.choices[0].finish_reason,
        "tool_call_finish_reason": tool_call_response.choices[0].finish_reason,
    }


@app.local_entrypoint()
def main():
    """
    Run two-step inference and display both reasoning and tool call.
    """
    print("=" * 80)
    print("QWEN 7B TWO-STEP INFERENCE")
    print("=" * 80)

    result = two_step_inference.remote()

    print("\n" + "=" * 80)
    print("STEP 1 - REASONING:")
    print("=" * 80)
    print(result["reasoning"])
    print()

    print("=" * 80)
    print("STEP 2 - TOOL CALL:")
    print("=" * 80)
    print(result["tool_call"])
    print()

    print("=" * 80)
    print("METRICS:")
    print("=" * 80)
    print(f"Reasoning time: {result['reasoning_time_seconds']:.2f}s")
    print(f"Tool call time: {result['tool_call_time_seconds']:.2f}s")
    print(f"Total time: {result['total_time_seconds']:.2f}s")
    print(f"Reasoning finish: {result['reasoning_finish_reason']}")
    print(f"Tool call finish: {result['tool_call_finish_reason']}")
    print("=" * 80)

