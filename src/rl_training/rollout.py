from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable

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
from src.models import AIModel, AgentRole
import art
from art.types import Messages
from openai.types.chat.chat_completion import Choice

import time

load_dotenv()

# Configure logging - suppress noisy HTTP logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


MAX_RETRIES = 3
MAX_TURNS = 100  # Prevent infinite games

# Default training completition models
DEFAULT_TRAINING_MODEL_SETUP = [
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_MINI,
    AIModel.OPENAI_GPT_5_MINI,
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


def _serialize_messages_and_choices(
    messages_and_choices: list[Any],
) -> list[Any]:
    serialized: list[Any] = []
    for item in messages_and_choices:
        if isinstance(item, dict):
            serialized.append(item)
            continue

        # Try common serialization hooks before falling back to repr
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                serialized.append(model_dump())
                continue
            except Exception:
                pass

        model_dump_json = getattr(item, "model_dump_json", None)
        if callable(model_dump_json):
            try:
                serialized.append(json.loads(model_dump_json()))
                continue
            except Exception:
                pass

        serialized.append({"repr": repr(item)})
    return serialized


@weave.op(tracing_sample_rate=1.0)
def log_final_trajectory_state(
    game_id: str, messages_and_choices: list[Any], reward: float
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "reward": reward,
        "messages_and_choices": _serialize_messages_and_choices(messages_and_choices),
    }


@weave.op(tracing_sample_rate=1.0)
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

            # Extract tool call FIRST (before adding to trajectory)
            tool_call_result = extract_tool_call_from_choice(choice)

            if tool_call_result is None:
                raise ValueError("No valid tool call in model response")

            tool_name, arguments = tool_call_result

            # Only add to trajectory after we know it's valid
            trajectory.messages_and_choices.append(choice)

            if verbose:
                logger.debug(f"Tool: {tool_name}, Args: {arguments}")

            # Success! Return the parsed tool call
            return tool_name, arguments, num_retries

        except openai.LengthFinishReasonError as e:
            # Model hit token limit - this is fatal, don't retry
            logger.warning(f"Token limit hit: {e}")
            raise e

        except Exception as e:
            num_retries += 1
            logger.debug(f"Retry {num_retries}/{max_retries}: {e}")

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


def get_policy_role(engine_api: EngineAPI, game_id: str) -> str | None:
    """
    Get the role assigned to the policy being trained.
    
    Returns:
        Role name ("liberal", "fascist", or "hitler"), or None if not found
    """
    engine = engine_api.engines.get(game_id)
    if not engine:
        return None
    
    # Get the policy agent
    policy_agent = engine.agents_by_id.get(engine.policy_agent_id)
    if not policy_agent:
        return None
    
    return policy_agent.role.value


def _analyze_discard_behavior(trajectory: art.Trajectory, policy_role: str | None) -> None:
    """
    Analyze the policy agent's discard behavior during the game.
    
    Tracks when the policy agent discards their own team's cards when they have
    both liberal and fascist cards available (strategic discard).
    
    Args:
        trajectory: The completed trajectory
        policy_role: The role assigned to the policy agent (liberal/fascist/hitler)
    """
    if policy_role is None:
        return
    
    # Determine which card type is "own" for this role
    # Liberal discards liberal = own card
    # Fascist/Hitler discards fascist = own card
    own_card_type = "liberal" if policy_role == "liberal" else "fascist"
    
    messages_and_choices = trajectory.messages_and_choices
    
    # Iterate through messages to find discard prompts
    i = 0
    while i < len(messages_and_choices):
        item = messages_and_choices[i]
        
        # Look for user messages with discard prompts
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content", "")
            
            # Check for president discard prompt
            if "Cards:" in content and "Discard index (0-2)" in content:
                cards, discard_idx = _parse_discard_action(messages_and_choices, i)
                if cards is not None and discard_idx is not None and len(cards) == 3:
                    trajectory.metrics["discard_as_president_count"] += 1
                    # Check if both card types are available
                    if "liberal" in cards and "fascist" in cards:
                        discarded_card = cards[discard_idx]
                        if discarded_card == own_card_type:
                            trajectory.metrics["discard_as_president_own_card_count"] += 1
            
            # Check for chancellor discard prompt
            elif "Cards:" in content and "Play index (0-1)" in content:
                cards, play_idx = _parse_discard_action(messages_and_choices, i)
                if cards is not None and play_idx is not None and len(cards) == 2:
                    trajectory.metrics["discard_as_chancellor_count"] += 1
                    # Check if both card types are available
                    if "liberal" in cards and "fascist" in cards:
                        # Chancellor discards the card they DON'T play
                        discard_idx = 1 - play_idx
                        discarded_card = cards[discard_idx]
                        if discarded_card == own_card_type:
                            trajectory.metrics["discard_as_chancellor_own_card_count"] += 1
        
        i += 1


def _parse_discard_action(
    messages_and_choices: list, user_msg_idx: int
) -> tuple[list[str] | None, int | None]:
    """
    Parse cards and discard/play choice from a discard prompt.
    
    Args:
        messages_and_choices: List of messages and choices
        user_msg_idx: Index of the user message with the discard prompt
    
    Returns:
        Tuple of (cards_list, chosen_index) or (None, None) if parsing fails
    """
    import re
    
    user_msg = messages_and_choices[user_msg_idx]
    content = user_msg.get("content", "")
    
    # Extract cards from the prompt
    # Format: "Cards: [PolicyCard.LIBERAL, PolicyCard.FASCIST, ...]"
    cards_match = re.search(r"Cards: \[(.*?)\]", content)
    if not cards_match:
        return None, None
    
    cards_str = cards_match.group(1)
    # Parse card types - extract just "liberal" or "fascist"
    cards = []
    for card in cards_str.split(","):
        if "LIBERAL" in card.upper():
            cards.append("liberal")
        elif "FASCIST" in card.upper():
            cards.append("fascist")
    
    if not cards:
        return None, None
    
    # Find the next assistant message with tool call
    for j in range(user_msg_idx + 1, len(messages_and_choices)):
        item = messages_and_choices[j]
        
        # Check if it's a Choice object with tool calls
        if isinstance(item, Choice):
            if item.message.tool_calls:
                tool_call = item.message.tool_calls[0]
                try:
                    args = json.loads(tool_call.function.arguments)
                    # Look for card_index in the arguments
                    if "card_index" in args:
                        return cards, args["card_index"]
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Stop if we hit another user message (moved to next turn)
        if isinstance(item, dict) and item.get("role") == "user":
            break
    
    return None, None


def get_messages_from_trajectory(
    messages_and_choices: art.types.MessagesAndChoices,
) -> Messages:
    messages: Messages = []
    for item in messages_and_choices:
        # If it's a Choice, it has to be a tool call, so we format it
        if isinstance(item, Choice):
            # Use attribute access for Pydantic models
            if item.message.tool_calls:
                # Convert tool_calls to dicts (they're Pydantic models too)
                msg = {
                    "role": "assistant", 
                    "content": item.message.content or "", 
                    "tool_calls": [tc.model_dump() for tc in item.message.tool_calls]
                }
                messages.append(msg)
        elif isinstance(item, dict):
            messages.append(item)
        else:
            raise ValueError(f"Unsupported message type: {type(item)}")
    return messages


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
    trainable_fascist_prob: float = 0.6,
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
        trainable_fascist_prob: Probability trainable agent gets Fascist/Hitler role (default: 0.6)

    Returns:
        Trajectory containing the game history and reward
    """
    # Start timing the entire rollout
    rollout_start_time = time.perf_counter()
    
    # Initialize the game
    game_id = str(uuid.uuid4())
    # Time the engine create call
    create_start = time.perf_counter()
    model_input = await _engine_api.create(
        game_id=game_id,
        deck=Deck(),
        ai_models=DEFAULT_TRAINING_MODEL_SETUP,
        trainable_fascist_prob=trainable_fascist_prob,
    )
    trainable_role = _engine_api.get_trainable_agent_role(game_id)
    trainable_role_value = trainable_role.value if trainable_role else None
    trainable_is_fascist = trainable_role in {AgentRole.FASCIST, AgentRole.HITLER}
    policy_role = get_policy_role(_engine_api, game_id)

    trajectory = art.Trajectory(
        messages_and_choices=[],
        metadata={
            "step": step,
            "validation": is_validation,
            "game_id": game_id,
            "trainable_role": trainable_role_value,
        },
        reward=0,
        metrics={
            "num_turns": 0,
            "invalid_tool_calls": 0,
            "total_retries": 0,
            "trainable_fascist_start": 1 if trainable_is_fascist else 0,
            "engine_execute_time_ms": 0.0,
            "total_engine_time_ms": 0.0,
        },
        # tools=TOOLS,  # Store tool schemas in trajectory
    )

    num_turns = 0
    game_over = False
    messages_added_count = 0  # Track how many messages we've already added
    final_state_logged = False

    if verbose:
        logger.info(f"Starting rollout | step={step} | game={game_id[:8]}")

    # Helper to clean multi-modal content
    def clean_msg(msg: dict) -> dict:
        """Convert multi-modal message format to plain string format for training."""
        msg = msg.copy()
        content = msg.get("content", [])
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and content[0].get("type") == "text":
                msg["content"] = content[0].get("text", "")
        return msg

    # Main game loop
    while not game_over and num_turns < max_turns:
        num_turns += 1

        # Add only NEW messages to trajectory (avoid duplicates)
        # Skip assistant messages since we add Choice objects directly
        new_messages = model_input.messages[messages_added_count:]
        for msg in new_messages:
            if msg.get("role") != "assistant":
                trajectory.messages_and_choices.append(clean_msg(msg))
        
        messages_added_count = len(model_input.messages)

        # Check if we have a terminal state (game over)
        if model_input.terminal_state is not None:
            game_over = True
            terminal_state: TerminalState = model_input.terminal_state
            
            trajectory.reward = terminal_state.reward
            if terminal_state.winning_team:
                trajectory.metadata["winning_team"] = terminal_state.winning_team
            assert terminal_state.game_id == game_id
            break

        # At this point, we should have a tool_call target
        if model_input.tool_call is None:
            raise ValueError(
                f"Expected tool_call target at turn {num_turns}, but got None. "
                "This should only happen when terminal_state is not None."
            )

        # Get valid tool call from model (with retries)
        try:
            with weave.thread(thread_id=game_id):
                tool_name, arguments, num_retries = await get_completion_with_retries(
                    model=model,
                    messages=get_messages_from_trajectory(trajectory.messages_and_choices),
                    tool_target=model_input.tool_call,
                    trajectory=trajectory,
                    enable_thinking=enable_thinking,
                    verbose=verbose,
                )

            # Track retry metrics
            trajectory.metrics["total_retries"] += num_retries

            # Format for engine and execute (with timing)
            model_function_calling_json = format_tool_response_for_game_engine(
                tool_name, arguments
            )
            execute_start = time.perf_counter()
            model_input = await _engine_api.execute(
                game_id, ModelOutput(function_calling_json=model_function_calling_json)
            )
            execute_time_ms = (time.perf_counter() - execute_start) * 1000

            # Track engine timing metrics
            trajectory.metrics["engine_execute_time_ms"] += execute_time_ms
            trajectory.metrics["total_engine_time_ms"] += execute_time_ms

        except (openai.LengthFinishReasonError, EngineException):
            # Fatal errors - propagate up to @art.retry decorator
            raise

        except Exception as e:
            # Unexpected error - mark as invalid and end game with penalty
            trajectory.metrics["invalid_tool_calls"] += 1
            trajectory.metrics["total_retries"] += MAX_RETRIES
            trajectory.reward = -1.0
            trajectory.metrics["failed_on_invalid_tool"] = True
            logger.warning(f"Game {game_id[:8]} ended with error: {e}")
            game_over = True

    # Record final metrics
    trajectory.metrics["num_turns"] = num_turns
    trajectory.metrics["hit_max_turns"] = num_turns >= max_turns

    if num_turns >= max_turns and not game_over:
        trajectory.reward = -0.5
        trajectory.metrics["forced_termination"] = True

    # Count trajectory composition
    choice_count = sum(1 for item in trajectory.messages_and_choices if hasattr(item, 'message'))
    msg_count = len(trajectory.messages_and_choices) - choice_count

    # Determine outcome (reward=0 means trainable agent's team lost, not a draw)
    outcome = "WIN" if trajectory.reward > 0 else "LOSS"

    # Log clean game summary to console
    logger.info(
        f"Game {game_id[:8]} | {outcome} | "
        f"reward={trajectory.reward:.2f} | turns={num_turns} | "
        f"choices={choice_count} | messages={msg_count}"
    )

    if not final_state_logged:
        with weave.thread(thread_id=game_id):
            log_final_trajectory_state(
                game_id=game_id,
                messages_and_choices=trajectory.messages_and_choices,
                reward=trajectory.reward,
            )

    # Calculate rollout duration
    rollout_duration_seconds = time.perf_counter() - rollout_start_time
    trajectory.metrics["rollout_duration_seconds"] = rollout_duration_seconds
    
    # Analyze discard behavior for the policy agent
    _analyze_discard_behavior(trajectory, policy_role)

    return trajectory


# Per-game timeout (in seconds) - if a single game takes longer, it fails individually
GAME_TIMEOUT = 20 * 60  # 20 minutes per game


async def rollout_with_timeout(
    model: art.Model,
    step: int,
    is_validation: bool,
    verbose: bool = False,
    max_turns: int = MAX_TURNS,
    enable_thinking: bool = True,
    trainable_fascist_prob: float = 0.6,
    game_timeout: int = GAME_TIMEOUT,
) -> art.Trajectory:
    """
    Wrapper around rollout that adds a per-game timeout.
    
    If a game takes longer than game_timeout, it raises asyncio.TimeoutError
    which ART's gather will count as an exception (not losing other games).
    """
    return await asyncio.wait_for(
        rollout(
            model=model,
            step=step,
            is_validation=is_validation,
            verbose=verbose,
            max_turns=max_turns,
            enable_thinking=enable_thinking,
            trainable_fascist_prob=trainable_fascist_prob,
        ),
        timeout=game_timeout,
    )
