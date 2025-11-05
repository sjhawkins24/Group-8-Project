"""Test OpenRouter API connection"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key loaded: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")
print(f"API Key starts with: {api_key[:20] if api_key else 'None'}...")

if not api_key:
    print("\n❌ ERROR: No API key found!")
    exit(1)

print("\n✅ API key is set. Testing OpenRouter connection...")

try:
    from openai import OpenAI
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'test successful' if you can read this."}
        ],
        max_tokens=20,
    )
    
    print(f"✅ OpenRouter API test successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
