import json
import math

import openai
import requests
import weave
from dotenv import load_dotenv
from .utils import (
    WINNING_VALUE,
    apply_agent_move,
    check_game_finished,
    generate_game,
    max_cell_value,
    render_board,
    total_board_value,
)

import art

load_dotenv()

# Define the tool for making moves in 2048
MAKE_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "make_move",
        "description": (
            "Make a move in the 2048 game. Choose one of the four directions: "
            "'left', 'right', 'up', or 'down'. This will slide all tiles in that direction "
            "and combine matching tiles. After your move, a new tile (2 or 4) will be added "
            "to a random empty cell."
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

TOOLS = [MAKE_MOVE_TOOL]


@weave.op
@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(
    model: art.Model,
    step: int,
    is_validation: bool,
    verbose: bool = False,
    max_turns: int = 200,
    enable_thinking: bool = True,
) -> art.Trajectory:
    """
    Run a single 2048 game rollout.

    Args:
        model: The ART model to use
        step: Training step number
        is_validation: Whether this is a validation run
        verbose: Print debug information
        max_turns: Maximum number of moves before forcing game end
        enable_thinking: Enable internal thinking for Qwen3 models (default: True)
            When True, Qwen3-4B-Thinking models will generate explicit reasoning
            in <think></think> blocks before making tool calls.

    Returns:
        Trajectory containing the game history and reward
    """
    game = generate_game()

    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an excellent 2048 player. Your goal is to combine tiles "
                    "to reach the number 2048 (or higher). Use the make_move tool to "
                    "make your moves. Analyze the board carefully and choose the move "
                    "most likely to lead to combining cells and eventually reaching 2048. "
                    "Available moves are: 'left', 'right', 'up', 'down'."
                ),
            },
        ],
        metadata={
            "game_id": game["id"],
            "step": step,
            "validation": is_validation,
        },
        reward=0,
    )

    while True:
        # Check if we've hit max turns
        if move_number >= max_turns:
            if verbose:
                print(f"Max turns ({max_turns}) reached, ending game")
            break

        # Add the current board state as a user message
        board_state = render_board(game)
        trajectory.messages_and_choices.append(
            {"role": "user", "content": f"{board_state}"}
        )
        if verbose:
            print(f"Move {move_number + 1}:")
            print(board_state)

        async def get_completion():
            client = model.openai_client()
            
            # Build base parameters
            params = {
                "max_completion_tokens": 512,  # Increased for thinking content
                "messages": trajectory.messages(),
                "model": model.name,
                "tools": TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "make_move"}},  # Force the model to use make_move tool
            }
            
            # For Qwen3 models, enable internal thinking via chat_template_kwargs
            # This allows the model to reason before generating tool calls
            if "qwen" in model.name.lower():
                params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            
            return await client.chat.completions.create(**params)

        try:
            chat_completion = await get_completion()
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        choice = chat_completion.choices[0]
        message = choice.message
        
        # Add the assistant's message (with tool call) to the trajectory
        trajectory.messages_and_choices.append(choice)

        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        direction = args["direction"]

        if verbose:
            print(f"Tool call: make_move(direction='{direction}')")

        try:
            apply_agent_move(game, f"<move>{direction}</move>")
            move_number += 1
        except ValueError:
            trajectory.metrics["invalid_move"] = 1
            trajectory.reward = -1
            break

        trajectory.messages_and_choices.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Move '{direction}' applied."
        })

        # Check if game is finished
        if check_game_finished(game):
            trajectory.metrics["invalid_move"] = 0
            break

    max_value = max_cell_value(game)
    board_value = total_board_value(game)
    agent_won = max_value == WINNING_VALUE
    trajectory.metrics["max_value"] = max_value
    trajectory.metrics["board_value"] = board_value
    trajectory.metrics["num_moves"] = move_number
    trajectory.metrics["win"] = agent_won

    # try to get as close to the winning value as possible
    # otherwise, try to maximize number of high cells on board
    # but above all else: WIN THE GAME!
    if agent_won:
        # double reward if the agent wins
        trajectory.reward = 2
    else:
        # scale max value logarithmically between 0 for 2 and 1 for WINNING_VALUE
        max_value_reward = (math.log(max_value, 2) - 1) / (
            math.log(WINNING_VALUE, 2) - 1
        )
        # scale board value logarithmically between 0 for 2 * 16 and 1 for WINNING_VALUE * 16
        board_value_reward = (math.log(board_value, 2) - 1) / (
            math.log(WINNING_VALUE * 16, 2) - 1
        )
        # combine the two rewards, with max value having a higher weight
        trajectory.reward = max_value_reward + (board_value_reward * 0.2)

    return trajectory
