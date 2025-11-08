import modal
import time

app = modal.App("rl-secret-hitler-inference")

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
    def generate(self, prompt: str, max_new_tokens: int = 32768):
        start_time = time.perf_counter()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
        
        total_time = time.perf_counter() - start_time
        
        return {
            "thinking_content": thinking_content,
            "content": content,
            "generation_time_seconds": generation_time,
            "total_time_seconds": total_time,
        }


@app.local_entrypoint()
def main():
    prompt = "Give me a short introduction to large language model. Be brief in your thinking and reasoning."
    
    end_to_end_start = time.perf_counter()
    inference = InferenceModel()
    result = inference.generate.remote(prompt)
    end_to_end_time = time.perf_counter() - end_to_end_start
    
    print("=" * 60)
    print("THINKING CONTENT:")
    print("=" * 60)
    print(result["thinking_content"])
    print()
    print("=" * 60)
    print("CONTENT:")
    print("=" * 60)
    print(result["content"])
    print()
    print("=" * 60)
    print("TIMING:")
    print("=" * 60)
    print(f"Generation time (model.generate): {result['generation_time_seconds']:.3f}s")
    print(f"Total time (including tokenization): {result['total_time_seconds']:.3f}s")
    print(f"End-to-end time (including Modal overhead): {end_to_end_time:.3f}s")
    print("=" * 60)
