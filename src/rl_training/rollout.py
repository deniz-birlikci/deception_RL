import json
import uuid

import openai
import requests
import weave
from dotenv import load_dotenv

from src.engine.protocol import (
    ModelOutput,
    TerminalState,
    ToolCallTarget,
)
from src.engine.engine_api import EngineAPI
from src.engine.deck import Deck
from src.models import AIModel
import art
from art.types import Messages

load_dotenv()


MAX_RETRIES = 3
MAX_TURNS = 100  # Prevent infinite games

# Default training completition models
DEFAULT_TRAINING_MODEL_SETUP = [
    AIModel.OPENAI_GPT_5,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_NANO,
    AIModel.OPENAI_GPT_4_1,
    None,
]

# Global engine API instance
_engine_api = EngineAPI()


class EngineException(Exception):
    """
    Exception raised when the engine fails to get a valid tool call.

    This indicates that the model failed to produce a valid tool call
    after MAX_RETRIES attempts. This is used to distinguish engine-level
    failures from other exceptions that might be retried at the rollout level.
    """

    pass


async def get_completion_with_tools(
    model: art.Model,
    messages: Messages,
    tool_target: ToolCallTarget,
    max_completion_tokens: int = 2048,
    enable_thinking: bool = True,
) -> openai.ChatCompletion:
    """
    Get LLM completion with a specific tool calling enabled.

    This function forces the model to call a single specific tool by:
    1. Providing only that tool's schema in the tools list
    2. Using tool_choice to explicitly require that specific tool

    For Qwen3 models, enable_thinking controls whether the model uses internal
    reasoning before generating tool calls (via chat_template_kwargs).

    Args:
        model: The ART model to use
        messages: Conversation messages in OpenAI format
        tool_target: The specific tool the model must call (name + OpenAI schema)
        max_completion_tokens: Maximum tokens for completion
        enable_thinking: Enable internal thinking for Qwen3 models (default: True)

    Returns:
        ChatCompletion with the tool call response
    """
    client = model.openai_client()

    # Build the tools list with just the single target tool
    tools = [tool_target.openai_schema]

    # Force the model to use this specific tool
    tool_choice = {"type": "function", "function": {"name": tool_target.name}}

    # Build base parameters
    params = {
        "max_completion_tokens": max_completion_tokens,
        "messages": messages,
        "model": model.name,
        "tools": tools,
        "tool_choice": tool_choice,
    }

    # For Qwen3 models, enable internal thinking via chat_template_kwargs
    # This allows the model to reason before generating tool calls
    if "qwen" in model.name.lower():
        params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    return await client.chat.completions.create(**params)


def extract_tool_call_from_choice(choice) -> tuple[str, dict] | None:
    """
    Extract the first tool call from the model's choice.

    Returns:
        (tool_name, arguments_dict) or None if no valid tool call
    """
    if not choice.message.tool_calls:
        return None

    tool_call = choice.message.tool_calls[0]
    tool_name = tool_call.function.name

    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return None

    return tool_name, arguments


async def get_completion_with_retries(
    model: art.Model,
    messages: Messages,
    tool_target: ToolCallTarget,
    trajectory: art.Trajectory,
    max_completion_tokens: int = 2048,
    enable_thinking: bool = True,
    max_retries: int = MAX_RETRIES,
    verbose: bool = False,
) -> tuple[str, dict, int]:
    """
    Get LLM completion with retries and extract tool call.

    Handles:
    - Retry logic for transient errors
    - Fatal errors (LengthFinishReasonError) that bypass retries
    - Tool call extraction and validation

    Args:
        model: The ART model to use
        messages: Conversation messages
        tool_target: The specific tool to call
        trajectory: Trajectory for tracking duration and choices
        max_completion_tokens: Max tokens for completion
        enable_thinking: Enable thinking for Qwen models
        max_retries: Maximum retry attempts
        verbose: Print debug info

    Returns:
        Tuple of (tool_name, arguments, num_retries_used)

    Raises:
        openai.LengthFinishReasonError: Model hit token limit (fatal, no retry)
        EngineException: Max retries exceeded without valid tool call
    """
    num_retries = 0

    while num_retries < max_retries:
        try:
            # Get LLM response with tool calling
            async with trajectory.track_duration("llm_completion"):
                chat_completion = await get_completion_with_tools(
                    model=model,
                    messages=messages,
                    tool_target=tool_target,
                    max_completion_tokens=max_completion_tokens,
                    enable_thinking=enable_thinking,
                )

            choice = chat_completion.choices[0]
            trajectory.messages_and_choices.append(choice)

            if verbose:
                print(f"Model response: {choice.message.tool_calls}")

            # Extract tool call
            tool_call_result = extract_tool_call_from_choice(choice)

            if tool_call_result is None:
                raise ValueError("No valid tool call in model response")

            tool_name, arguments = tool_call_result

            if verbose:
                print(f"Tool: {tool_name}, Args: {arguments}")

            # Success! Return the parsed tool call
            return tool_name, arguments, num_retries

        except openai.LengthFinishReasonError as e:
            # Model hit token limit - this is fatal, don't retry
            if verbose:
                print(f"Length finish reason error: {e}")
            raise e

        except Exception as e:
            num_retries += 1

            if verbose:
                print(f"Error on attempt {num_retries}/{max_retries}: {e}")

            if num_retries >= max_retries:
                # Max retries exceeded - raise EngineException
                raise EngineException(
                    f"Failed after {max_retries} retries. Last error: {e}"
                )

    # Should never reach here, but for type safety
    raise EngineException(f"Failed after {max_retries} retries")


def format_tool_response_for_game_engine(tool_name: str, arguments: dict) -> str:
    """
    Format the tool call into a JSON string for the game engine to parse.

    The game engine expects:
    {
        "tool_name": "president-pick-chancellor",
        "arguments": {"agent_id": "..."}
    }
    """
    return json.dumps({"tool_name": tool_name, "arguments": arguments})


@weave.op
@art.retry(
    exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout, EngineException)
)
async def rollout(
    model: art.Model,
    step: int,
    is_validation: bool,
    verbose: bool = False,
    max_turns: int = MAX_TURNS,
    enable_thinking: bool = True,
) -> art.Trajectory:
    """
    Run a single Secret Hitler game rollout.

    The model plays all 5 agents in the game. Each agent gets appropriate
    context (role, private info, public events) and must use tool calling
    to take actions.

    Args:
        model: The ART model to use
        step: Training step number
        is_validation: Whether this is a validation run
        verbose: Print debug information
        max_turns: Maximum number of turns before forcing game end
        enable_thinking: Enable internal thinking for Qwen3 models (default: True)

    Returns:
        Trajectory containing the game history and reward
    """
    # Initialize the game
    game_id = str(uuid.uuid4())
    model_input = await _engine_api.create(
        game_id=game_id,
        deck=Deck(),
        ai_models=DEFAULT_TRAINING_MODEL_SETUP,
    )

    trajectory = art.Trajectory(
        messages_and_choices=[],
        metadata={
            "step": step,
            "validation": is_validation,
            "game_id": game_id,
        },
        reward=0,
        metrics={
            "num_turns": 0,
            "invalid_tool_calls": 0,
            "total_retries": 0,
        },
        # tools=TOOLS,  # Store tool schemas in trajectory
    )

    num_turns = 0
    game_over = False

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Starting Secret Hitler Rollout (Step {step})")
        print(f"{'=' * 60}\n")

    # Main game loop
    while not game_over and num_turns < max_turns:
        num_turns += 1

        if verbose:
            print(f"\n--- Turn {num_turns} ---")
            print(f"Messages to model ({len(model_input.messages)} messages)")

        # Add the current game state messages to the trajectory
        for msg in model_input.messages:
            trajectory.messages_and_choices.append(msg)

        # Check if we have a terminal state (game over)
        if model_input.terminal_state is not None:
            game_over = True
            terminal_state: TerminalState = model_input.terminal_state
            trajectory.reward = terminal_state.reward
            # Validate game_id matches (sanity check)
            assert terminal_state.game_id == game_id, f"Terminal state game_id mismatch: {terminal_state.game_id} != {game_id}"

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Game Over! Reward: {terminal_state.reward}")
                print(f"{'=' * 60}\n")
            break

        # At this point, we should have a tool_call target
        if model_input.tool_call is None:
            raise ValueError(
                f"Expected tool_call target at turn {num_turns}, but got None. "
                "This should only happen when terminal_state is not None."
            )

        # Get valid tool call from model (with retries)
        try:
            tool_name, arguments, num_retries = await get_completion_with_retries(
                model=model,
                messages=model_input.messages,
                tool_target=model_input.tool_call,
                trajectory=trajectory,
                enable_thinking=enable_thinking,
                verbose=verbose,
            )

            # Track retry metrics
            trajectory.metrics["total_retries"] += num_retries

            # Format for engine and execute
            model_function_calling_json = format_tool_response_for_game_engine(
                tool_name, arguments
            )
            model_input = await _engine_api.execute(
                game_id, ModelOutput(function_calling_json=model_function_calling_json)
            )

        except (openai.LengthFinishReasonError, EngineException):
            # Fatal errors - propagate up to @art.retry decorator
            # These will trigger a full rollout retry
            raise

        except Exception as e:
            # Unexpected error - mark as invalid and end game with penalty
            trajectory.metrics["invalid_tool_calls"] += 1
            trajectory.metrics["total_retries"] += MAX_RETRIES
            trajectory.reward = -1.0
            trajectory.metrics["failed_on_invalid_tool"] = True

            if verbose:
                print(f"Unexpected error: {e}")
                print("Ending game with penalty.")

            game_over = True

    # Record final metrics
    trajectory.metrics["num_turns"] = num_turns
    trajectory.metrics["hit_max_turns"] = num_turns >= max_turns

    if num_turns >= max_turns and not game_over:
        # Penalty for games that don't finish
        trajectory.reward = -0.5
        trajectory.metrics["forced_termination"] = True

        if verbose:
            print(
                f"Game hit max turns ({max_turns}) without finishing. Penalty applied."
            )

    if verbose:
        print("\nFinal Trajectory Stats:")
        print(f"  Turns: {num_turns}")
        print(f"  Reward: {trajectory.reward}")
        print(f"  Invalid tool calls: {trajectory.metrics['invalid_tool_calls']}")
        print(f"  Total retries: {trajectory.metrics['total_retries']}")

    return trajectory
