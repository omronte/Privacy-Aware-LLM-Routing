import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.openai_compatible import get_external_client

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = os.getenv(
    "REMOTE_MODEL",
    "openai/gpt-oss-120b",
)

print("=" * 60)
print("GROQ REMOTE PROVIDER TEST")
print("=" * 60)

external_client = get_external_client(MODEL)
print(
    "External provider configured:",
    "YES" if external_client else "NO",
)
print("HF token found:", "YES" if HF_TOKEN else "NO")
print("Model:", MODEL)

if external_client:
    client = external_client
elif not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")
else:
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
