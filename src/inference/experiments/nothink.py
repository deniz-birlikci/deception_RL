"""
Minimal Modal entrypoint to inspect raw Qwen3 reasoning + tool-call output.

This uses ART to spin up a vLLM-backed Qwen3-4B-Thinking server, sends a single
prompt, and prints the untouched response payload so we can examine the tokens
and native tool-call formatting ourselves.
"""

import json
import modal
import time

app = modal.App("qwen3-raw-vllm-response")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("procps")
    .pip_install(
        "openpipe-art[backend]",
        "openai>=1.65.5",
    )
)

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    secrets=[modal.Secret.from_name("art-secrets")],
)
async def fetch_raw_response(enable_thinking: bool = True):
    """
    Launch the Qwen3 model via ART and return the raw OpenAI-compatible response.
    """
    import art
    from art.local import LocalBackend

    start_time = time.perf_counter()

    model = art.TrainableModel(
        name="qwen3-raw-demo",
        project="nothink",
        base_model=MODEL_NAME,
    )

    # Reuse the same engine overrides we use elsewhere – keep reasoning flags,
    # but let us inspect the raw payload instead of parsing anything.
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=8192,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
            additional_config={
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
                "reasoning_backend": "qwen3",
            },
        ),
    )

    backend = LocalBackend()
    await model.register(backend)
    await backend._get_service(model)  # type: ignore[attr-defined]

    prompt = (
        "Consider the 2048 board below.\n"
        " 64| 32| 16|  8\n"
        " 32| 64| 32| 16\n"
        " 16| 32| 64| 32\n"
        "  8| 16| 32|128\n\n"
        "Produce two sequential sections:\n"
        "1. A <think>...</think> block containing at least 1024 tokens of detailed reasoning, "
        "exploring multiple candidate moves, future board states, and long-horizon plans. "
        "Do not close </think> until you have thoroughly reasoned past the 1024-token mark.\n"
        "2. Immediately after </think>, output exactly one tag of the form "
        "<tool_call>{\"move\": \"left|right|up|down\"}</tool_call> representing your final decision. "
        "Include no additional text after the tool call."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous strategist. Obey formatting instructions strictly: "
                "first an extended <think> block with private reasoning, then one <tool_call> JSON payload "
                "containing the chosen move. Do not emit extra prose outside these tags."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    client = model.openai_client()

    params = {
        "model": model.name,
        "messages": messages,
        "max_completion_tokens": 2048,
    }

    if "qwen3" in MODEL_NAME.lower():
        if enable_thinking:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        else:
            params["extra_body"] = {}
    elif enable_thinking:
        params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    response = await client.chat.completions.create(**params)

    return {
        "raw_response": response.model_dump() if hasattr(response, "model_dump") else json.loads(response.json()),
        "generation_time_seconds": time.perf_counter() - start_time,
        "enable_thinking": enable_thinking,
    }


@app.local_entrypoint()
def main(enable_thinking: bool = True):
    """
    Run the raw response fetch and pretty-print the payload for inspection.
    """
    result = fetch_raw_response.remote(enable_thinking)

    print("=" * 80)
    print("RAW RESPONSE")
    print("=" * 80)
    print(json.dumps(result["raw_response"], indent=2, default=str))
    print("=" * 80)
    print(f"Thinking enabled: {result['enable_thinking']}")
    print(f"Generation time: {result['generation_time_seconds']:.2f}s")

