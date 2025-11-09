"""
Test script to demonstrate Qwen3-4B-Thinking-2507 with tool calls for 2048.

This script shows how the model generates reasoning before making tool calls.

Usage:
    python example_2048/test_thinking.py
"""

import json


def demonstrate_thinking_flow():
    """
    Demonstrates how Qwen3-4B-Thinking works with tool calls.
    
    This is a mock example showing the expected flow when enable_thinking=True.
    """
    print("=" * 80)
    print("QWEN3-4B-THINKING-2507 WITH TOOL CALLS - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Example 1: Simple move
    print("EXAMPLE 1: Simple Strategic Move")
    print("-" * 80)
    print("Board State:")
    print("  2|  4|  8| 16")
    print(" 32| 64|128|256")
    print("  _|  _|  _|  _")
    print("  _|  _|  _|  _")
    print()
    
    print("Model Output with enable_thinking=True:")
    print()
    print("<think>")
    print("Looking at the board, I can see there's a nice sequence building up in")
    print("the second row: 32, 64, 128, 256. The top row also has a sequence: 2, 4, 8, 16.")
    print()
    print("If I move down, the bottom two rows are empty, so nothing will combine.")
    print("If I move up, same issue - no combinations.")
    print()
    print("If I move left, the top row stays the same (already against the left).")
    print("The second row also stays the same.")
    print()
    print("If I move right, I can potentially set up for future combinations.")
    print("But actually, I should move down to fill the bottom and prepare for")
    print("horizontal combinations. Let me move down.")
    print("</think>")
    print()
    print("Tool Call:")
    print(json.dumps({
        "name": "make_move",
        "arguments": {
            "direction": "down"
        }
    }, indent=2))
    print()
    
    # Example 2: Complex reasoning
    print("=" * 80)
    print("EXAMPLE 2: Complex Strategic Decision")
    print("-" * 80)
    print("Board State:")
    print(" 64| 32| 16|  8")
    print(" 32| 64| 32| 16")
    print(" 16| 32| 64| 32")
    print("  8| 16| 32| 64")
    print()
    
    print("Model Output with enable_thinking=True:")
    print()
    print("<think>")
    print("This is a tricky board state. I have many opportunities for combinations.")
    print()
    print("Option 1 - Move LEFT:")
    print("  - Row 1: 64, 32, 16, 8 → No direct combinations")
    print("  - Row 2: 32, 64, 32, 16 → Could combine the two 32s")
    print("  - Row 3: 16, 32, 64, 32 → Could combine the two 32s")
    print("  - Row 4: 8, 16, 32, 64 → No direct combinations")
    print("  Total: 2 combinations = Good")
    print()
    print("Option 2 - Move RIGHT:")
    print("  - Similar to left, but combinations happen on the right side")
    print("  Total: 2 combinations = Good")
    print()
    print("Option 3 - Move UP:")
    print("  - Col 1: 64, 32, 16, 8 → No combinations")
    print("  - Col 2: 32, 64, 32, 16 → The two 32s combine")
    print("  - Col 3: 16, 32, 64, 32 → The two 32s combine")
    print("  - Col 4: 8, 16, 32, 64 → No combinations")
    print("  Total: 2 combinations = Good")
    print()
    print("Option 4 - Move DOWN:")
    print("  - Similar analysis to UP")
    print("  Total: 2 combinations = Good")
    print()
    print("All moves give 2 combinations. I should choose based on positioning.")
    print("Moving UP will keep my higher values (64s) toward the top, which is")
    print("generally a good strategy - keep big numbers in one area.")
    print("I'll move UP.")
    print("</think>")
    print()
    print("Tool Call:")
    print(json.dumps({
        "name": "make_move",
        "arguments": {
            "direction": "up"
        }
    }, indent=2))
    print()
    
    # Comparison with non-thinking
    print("=" * 80)
    print("COMPARISON: Without enable_thinking (enable_thinking=False)")
    print("-" * 80)
    print("Board State: [Same as Example 1]")
    print()
    print("Model Output with enable_thinking=False:")
    print()
    print("Tool Call:")
    print(json.dumps({
        "name": "make_move",
        "arguments": {
            "direction": "down"
        }
    }, indent=2))
    print()
    print("Note: No explicit reasoning is shown. The model still 'thinks' internally,")
    print("but doesn't generate the <think></think> blocks.")
    print()
    
    # Benefits summary
    print("=" * 80)
    print("BENEFITS OF enable_thinking=True FOR RL TRAINING")
    print("=" * 80)
    print()
    print("1. INTERPRETABILITY")
    print("   - See exactly why the model chose each move")
    print("   - Debug poor decisions during training")
    print("   - Understand learned strategies")
    print()
    print("2. BETTER LEARNING")
    print("   - Model learns to generate coherent reasoning")
    print("   - Reasoning chains become more sophisticated over training")
    print("   - Potentially faster convergence to good policies")
    print()
    print("3. STRATEGIC THINKING")
    print("   - Model learns to look ahead (if I move X, then Y...)")
    print("   - Develops understanding of game mechanics")
    print("   - Can articulate strategic principles")
    print()
    print("4. ROBUSTNESS")
    print("   - Explicit reasoning reduces random/erratic moves")
    print("   - More consistent decision-making")
    print("   - Better generalization to new board states")
    print()
    print("=" * 80)
    print()
    
    # Implementation details
    print("=" * 80)
    print("IMPLEMENTATION DETAILS")
    print("=" * 80)
    print()
    print("In rollout.py, we enable thinking like this:")
    print()
    print("```python")
    print("params = {")
    print("    'max_completion_tokens': 512,  # Increased for thinking")
    print("    'messages': messages,")
    print("    'model': model.name,")
    print("    'tools': TOOLS,")
    print("    'tool_choice': {'type': 'function', 'function': {'name': 'make_move'}},")
    print("}")
    print()
    print("# For Qwen3 models, enable thinking via chat_template_kwargs")
    print("if 'qwen' in model.name.lower():")
    print("    params['extra_body'] = {")
    print("        'chat_template_kwargs': {'enable_thinking': enable_thinking}")
    print("    }")
    print()
    print("response = await client.chat.completions.create(**params)")
    print("```")
    print()
    print("The key is passing chat_template_kwargs through extra_body to vLLM.")
    print("This tells the model to generate <think></think> blocks before tool calls.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_thinking_flow()

