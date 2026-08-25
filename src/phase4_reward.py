
import re
import torch
import torch.nn.functional as F

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_judge_client = None

def get_judge_client(hf_token):

    from huggingface_hub import InferenceClient

    global _judge_client

    if _judge_client is None:

        _judge_client = InferenceClient(
            model=JUDGE_MODEL,
            provider="auto",
            token=hf_token,
            timeout=60
        )

    return _judge_client


DEFAULT_LAMBDA_PRIVACY = 5.0

TASK_GAIN_THRESHOLD = 0.55



def calculate_privacy_leak(
    pii_entities,
    remote_chunks
):
    """
    Calculate the fraction of PII entities exposed
    to the remote model.

    PrivacyLeak =
        number of PII units appearing in remote chunks
        ------------------------------------------------
                    total number of PII units

    Matching is case-insensitive substring matching.

    Returns:
        float in [0, 1]
    """

    if not pii_entities:
        return 0.0

    if not remote_chunks:
        return 0.0

    remote_text = " ".join(
        str(chunk)
        for chunk in remote_chunks
    ).lower()

    pii_entities = [
        str(entity).strip()
        for entity in pii_entities
        if str(entity).strip()
    ]

    if not pii_entities:
        return 0.0

    leaked_count = 0

    for entity in pii_entities:

        if entity.lower() in remote_text:
            leaked_count += 1

    privacy_leak = (
        leaked_count /
        len(pii_entities)
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                privacy_leak
            )
        )
    )



def _get_sentence_transformer():
    """
    Lazily load MiniLM for response evaluation.

    The model is loaded only once.
    """

    global _similarity_model

    try:
        return _similarity_model
    except NameError:
        _similarity_model = None

    if _similarity_model is None:

        from sentence_transformers import SentenceTransformer

        _similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return _similarity_model



def evaluate_task_gain(
    query,
    gold_response,
    generated_response,
    hf_token
):
    """
    Evaluate how well the generated response answers the query.

    Returns:
        0.0 -> incorrect / irrelevant
        0.5 -> partially correct
        1.0 -> correct and useful
    """

    if not generated_response or not generated_response.strip():
        print("LLM judge received empty generated response.")
        return 0.0

    if not gold_response or not gold_response.strip():
        print("LLM judge received empty gold response.")
        return 0.0

    client = get_judge_client(hf_token)

    prompt = f"""
Evaluate the following generated answer.

QUERY:
{query}

REFERENCE ANSWER:
{gold_response}

GENERATED ANSWER:
{generated_response}

Scoring:

2 = The generated answer is correct, relevant, and sufficiently complete.
1 = The generated answer is partially correct or incomplete.
0 = The generated answer is incorrect or irrelevant.

IMPORTANT:
The generated answer does NOT need to use the same wording as the
reference answer.

Return ONLY:
0
1
or
2
"""

    try:

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict but fair answer evaluator. "
                        "Evaluate semantic correctness, not exact wording. "
                        "Return ONLY one digit: 0, 1, or 2."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=10,
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )

        if not response.choices:
            print("LLM judge returned no choices.")
            return 0.0

        message = response.choices[0].message


        content = getattr(message, "content", None)

        if content is None:
            content = ""

        content = str(content).strip()
        content = content.replace("\\r\\n", "\n").replace("\\n", "\n")

        first_line = next(
            (line.strip() for line in content.splitlines() if line.strip()),
            "",
        )
        match = re.fullmatch(r"([012])", first_line)

        if not match:
            match = re.match(r"([012])(?:\s|$)", first_line)

        if not match:
            print(f"LLM judge returned invalid content: {content!r}")
            return 0.0

        result = match.group(1)

        print(f"LLM JUDGE RESULT: {result!r}")

        score = int(result)

        if score == 2:
            return 1.0

        if score == 1:
            return 0.5

        return 0.0

    except Exception as e:

        if "402" in str(e) or "Payment Required" in str(e):
            raise RuntimeError(
                "Hugging Face inference credits are exhausted. "
                "Restore provider credits before continuing PPO training."
            ) from e

        print(
            f"LLM judge failed: {e}"
        )

        return 0.0



def calculate_reward(
    task_gain,
    privacy_leak,
    routing_compliance=0.0,
    lambda_privacy=DEFAULT_LAMBDA_PRIVACY,
):

    ROUTING_BONUS = 0.20

    reward = (
        float(task_gain)
        - float(lambda_privacy) * (float(privacy_leak) ** 2)
        + ROUTING_BONUS * float(routing_compliance)
    )

    return float(reward)



def print_reward_scenarios(
    lambda_privacy=DEFAULT_LAMBDA_PRIVACY
):
    """
    Print three simple reward sanity checks.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 4 REWARD SANITY CHECK"
    )

    print(
        "=" * 60
    )

    scenarios = [
        (
            "A - 100% Local",
            0.0,
            0.0,
            1.0,
        ),
        (
            "B - 100% Remote",
            1.0,
            1.0,
            0.0,
        ),
        (
            "C - Partial Leak",
            1.0,
            0.25,
            0.5,
        )
    ]

    for name, task_gain, privacy_leak, routing_compliance in scenarios:

        reward = calculate_reward(
            task_gain=task_gain,
            privacy_leak=privacy_leak,
            routing_compliance=routing_compliance,
            lambda_privacy=lambda_privacy,
        )

        print(
            f"{name:<22}"
            f"TaskGain={task_gain:.2f} | "
            f"Leak={privacy_leak:.2f} | "
            f"Reward={reward:.4f} | "
            f"Routing Compliance={routing_compliance:.4f}"
        )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    print_reward_scenarios()
