import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

print("=" * 60)
print("HUGGING FACE REMOTE TEST")
print("=" * 60)

print("HF token found:", "YES" if HF_TOKEN else "NO")
print("Model:", MODEL)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

client = InferenceClient(
    model=MODEL,
    provider="auto",
    token=HF_TOKEN,
    timeout=60
)

try:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain supervised learning in one sentence."
            }
        ],
        max_tokens=200,
        temperature=0.0
    )

    message = response.choices[0].message

    print("\nSTATUS: SUCCESS")
    print("CONTENT:", repr(message.content))
    print("REASONING:", repr(getattr(message, "reasoning_content", None)))
    print("FINISH:", response.choices[0].finish_reason)

except Exception as e:
    print("\nSTATUS: FAIL")
    print(type(e).__name__)
    print(e)
