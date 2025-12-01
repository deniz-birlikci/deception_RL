import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_gpt5_model():
    """
    Test GPT-5 model and time the end-to-end response.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    test_prompt = """
    You are playing Secret Impostor, a social deduction game. You've been assigned the role of a Crewmate.
    
    The current situation:
    - 5 players total
    - 2 sabotage protocols and 1 security protocol have been resolved
    - Player A was just appointed First Mate
    - Player B (the Captain) must now choose 2 event cards from 3 draws to pass to Player A
    - The 3 cards are: Sabotage, Sabotage, Security
    
    As a Crewmate player observing this, what should you be thinking about? What are the key considerations for this moment in the game? Analyze the strategic implications and what you should watch for.
    """
    
    print("=" * 60)
    print("Testing GPT-5 Model")
    print("=" * 60)
    print(f"Test prompt: {test_prompt[:100]}...")
    print("\nSending request...")
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": test_prompt}],
            max_completion_tokens=3000
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        ai_response = response.choices[0].message.content
        
        print(f"\n✅ Response received!")
        print(f"⏱️  End-to-end time: {elapsed_time:.2f} seconds")
        print(f"📊 Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        print("\n" + "=" * 60)
        print("AI RESPONSE:")
        print("=" * 60)
        print(ai_response)
        print("=" * 60)

        return elapsed_time, ai_response

    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n❌ Error after {elapsed_time:.2f} seconds:")
        print(f"Error: {str(e)}")
        
        if "api_key" in str(e).lower():
            print("\n💡 Tip: Set your OpenAI API key with:")
            print("export OPENAI_API_KEY='your-key-here'")
        
        return elapsed_time, None

if __name__ == "__main__":
    print("Starting GPT-5 Model Test...")
    test_gpt5_model()
