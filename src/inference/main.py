"""
Qwen3 reasoning + tool calling inference using ART.

Key setup:
- reasoning_backend="qwen3" - Parses <think> tokens → reasoning_content
- tool_call_parser="hermes" - Handles tool call parsing
- chat_template_kwargs={"enable_thinking": True} - Enables thinking in prompts

Usage:
    modal run src/inference/main.py
"""

import modal
import time
import json
from pathlib import Path

app = modal.App("qwen3-reasoning-tool-calling-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("procps")  # Provides pkill command needed by ART backend
    .pip_install(
        "openpipe-art[backend]",
        "openai>=1.65.5",
    )
)

model_name = "Qwen/Qwen3-4B-Thinking-2507"

# Tool definition for 2048 game
MAKE_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "make_move",
        "description": (
            "Make a move in the 2048 game. Choose one of the four directions: "
            "'left', 'right', 'up', or 'down'. This will slide all tiles in that direction "
            "and combine matching tiles."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down"],
                    "description": "The direction to move all tiles on the board.",
                }
            },
            "required": ["direction"],
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
async def inference_with_reasoning_and_tools(
    board_state: str,
    enable_thinking: bool = True,
    force_tool: bool = True,
):
    """
    Run Qwen3 inference with reasoning + tool calling.
    
    Returns dict with: reasoning_content, tool_call, timing metrics
    """
    import art
    from art.local import LocalBackend
    from art.utils.output_dirs import get_model_dir
    import os
    
    start_time = time.perf_counter()
    
    # Try setting vLLM config via environment variables (most reliable method)
    os.environ["VLLM_REASONING_BACKEND"] = "qwen3"
    os.environ["VLLM_TOOL_CALL_PARSER"] = "hermes"
    os.environ["VLLM_ENABLE_AUTO_TOOL_CHOICE"] = "1"
    print("[debug] Set vLLM env vars: REASONING_BACKEND=qwen3, TOOL_CALL_PARSER=hermes")
    
    # Configure model with vLLM reasoning + tool calling
    model = art.TrainableModel(
        name="qwen3-reasoning-inference",
        project="inference-test",
        base_model=model_name,
    )
    
    # Inspect what EngineArgs actually accepts
    import inspect
    engine_args_sig = inspect.signature(art.dev.EngineArgs.__init__)
    print("[debug] EngineArgs available params:", list(engine_args_sig.parameters.keys())[:20])  # First 20
    
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(max_seq_length=8192),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
        ),
    )
    print("[debug] initial engine_args dict:", model._internal_config["engine_args"])
    
    # Load model
    backend = LocalBackend()
    await model.register(backend)
    service_config = getattr(backend, "_service_config", None)
    if service_config:
        print("[debug] service_config.engine_args:", service_config.get("engine_args"))
    else:
        print("[debug] service_config unavailable on backend")
    await backend._get_service(model)  # type: ignore[attr-defined]
    print("✓ Model loaded")

    # Surface recent vLLM log lines to Modal logs
    try:
        output_dir = Path(get_model_dir(model=model, art_path=backend._path))
        log_path = output_dir / "logs" / "vllm.log"
        if log_path.exists():
            log_lines = log_path.read_text().splitlines()
            tail = "\n".join(log_lines[-100:])
            print("\n[vllm.log tail]\n" + tail + "\n[/vllm.log tail]\n")
        else:
            print(f"[debug] vLLM log not found at {log_path}")
    except Exception as err:
        print(f"[debug] failed to read vLLM log: {err}")
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert 2048 planner. Think step-by-step about the board, "
                "then call the `make_move` tool with your chosen direction."
            ),
        },
        {
            "role": "user",
            "content": f"Board:\n{board_state}\n\nThink internally, then make your move.",
        },
    ]
    
    # Build API parameters
    client = model.openai_client()
    params = {
        "model": model.name,
        "messages": messages,
        "tools": [MAKE_MOVE_TOOL],
        "max_completion_tokens": 512,
    }
    
    if force_tool:
        params["tool_choice"] = {"type": "function", "function": {"name": "make_move"}}
    
    if enable_thinking:
        params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    
    # Generate
    generation_start = time.perf_counter()
    response = await client.chat.completions.create(**params)
    generation_time = time.perf_counter() - generation_start
    
    # Extract results
    choice = response.choices[0]
    message = choice.message
    
    reasoning_content = getattr(message, 'reasoning_content', None)
    
    tool_call = None
    if message.tool_calls:
        tc = message.tool_calls[0]
        tool_call = {
            "name": tc.function.name,
            "arguments": json.loads(tc.function.arguments),
        }
    
    return {
        "reasoning_content": reasoning_content,
        "tool_call": tool_call,
        "generation_time_seconds": generation_time,
        "total_time_seconds": time.perf_counter() - start_time,
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if hasattr(response, 'usage') else None,
    }


@app.local_entrypoint()
def main(enable_thinking: bool = True, force_tool: bool = True):
    """
    Test Qwen3 reasoning + tool calling inference.
    
    Usage:
        modal run src/inference/main.py
        modal run src/inference/main.py --enable-thinking=False
    """
    board_state = """
 64| 32| 16|  8
 32| 64| 32| 16
 16| 32| 64| 32
  8| 16| 32|128
"""
    
    print("=" * 80)
    print("QWEN3 REASONING + TOOL CALLING TEST")
    print("=" * 80)
    print(f"Thinking enabled: {enable_thinking}")
    print(f"Force tool choice: {force_tool}")
    print("\nBoard State:")
    print(board_state)
    print("=" * 80)
    
    result = inference_with_reasoning_and_tools.remote(board_state, enable_thinking, force_tool)
    
    if result["reasoning_content"]:
        print("\n" + "=" * 80)
        print("REASONING CONTENT:")
        print("=" * 80)
        print(result["reasoning_content"])
    
    print("\n" + "=" * 80)
    print("TOOL CALL:")
    print("=" * 80)
    if result["tool_call"]:
        print(f"Function: {result['tool_call']['name']}")
        print(f"Arguments: {json.dumps(result['tool_call']['arguments'], indent=2)}")
        print(f"\n✓ Success: {result['tool_call']['name']}(direction='{result['tool_call']['arguments']['direction']}')")
    else:
        print("✗ No tool call generated")
    
    print("\n" + "=" * 80)
    print("METRICS:")
    print("=" * 80)
    print(f"Generation time: {result['generation_time_seconds']:.2f}s")
    print(f"Total time: {result['total_time_seconds']:.2f}s")
    print(f"Finish reason: {result['finish_reason']}")
    if result.get('usage'):
        print(f"Tokens: {result['usage']}")
    print("=" * 80)
