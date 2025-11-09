import modal
import time
import json

app = modal.App("rl-secret-hitler-inference-with-tools")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "sentencepiece",
    )
)

model_name = "Qwen/Qwen3-4B-Thinking-2507"

# Simple tool definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number to add.",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number to add.",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]

def add_numbers(a: float, b: float) -> float:
    """Execute the add_numbers tool."""
    return a + b

@app.cls(
    image=image,
    gpu="H100",
    scaledown_window=300,
)
class InferenceModel:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    @modal.method()
    def generate(self, prompt: str, tools: list = None, max_new_tokens: int = 32768):
        start_time = time.perf_counter()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template with tools if provided
        template_kwargs = {"add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
            template_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **template_kwargs
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generation_start = time.perf_counter()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generation_time = time.perf_counter() - generation_start
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # Try to parse tool calls from content if present
        tool_calls = []
        if tools and content:
            # Qwen wraps tool calls in <tool_call> tags
            import re
            tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            matches = re.findall(tool_call_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if "name" in parsed and "arguments" in parsed:
                        tool_calls.append(parsed)
                except (json.JSONDecodeError, KeyError):
                    pass
        
        total_time = time.perf_counter() - start_time
        
        return {
            "thinking_content": thinking_content,
            "content": content,
            "tool_calls": tool_calls,
            "generation_time_seconds": generation_time,
            "total_time_seconds": total_time,
        }


@app.local_entrypoint()
def main():
    prompt = "What is 42 plus 17? Think about it step by step, then use the add_numbers tool to compute the result."
    
    end_to_end_start = time.perf_counter()
    inference = InferenceModel()
    result = inference.generate.remote(prompt, tools=TOOLS)
    end_to_end_time = time.perf_counter() - end_to_end_start
    
    print("=" * 80)
    print("THINKING CONTENT (Private Reasoning):")
    print("=" * 80)
    print(result["thinking_content"])
    print()
    print("=" * 80)
    print("CONTENT (Public Response):")
    print("=" * 80)
    print(result["content"])
    print()
    
    # Execute tool calls if present
    if result["tool_calls"]:
        print("=" * 80)
        print("TOOL CALLS:")
        print("=" * 80)
        for tool_call in result["tool_calls"]:
            print(f"Function: {tool_call['name']}")
            print(f"Arguments: {json.dumps(tool_call['arguments'], indent=2)}")
            
            # Execute the tool
            if tool_call['name'] == 'add_numbers':
                args = tool_call['arguments']
                result_value = add_numbers(args['a'], args['b'])
                print(f"Result: {result_value}")
            print()
    else:
        print("=" * 80)
        print("NOTE: No tool calls detected in response")
        print("=" * 80)
        print()
    
    print("=" * 80)
    print("TIMING:")
    print("=" * 80)
    print(f"Generation time (model.generate): {result['generation_time_seconds']:.3f}s")
    print(f"Total time (including tokenization): {result['total_time_seconds']:.3f}s")
    print(f"End-to-end time (including Modal overhead): {end_to_end_time:.3f}s")
    print("=" * 80)