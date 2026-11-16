
from __future__ import annotations

import torch
import os

try:
    from .privacy import contains_pii, redact_pii
except ImportError:
    from privacy import contains_pii, redact_pii

try:
    from .phase4_reward import (
        calculate_privacy_leak,
        calculate_reward,
        evaluate_task_gain,
    )
    from .phase2_preprocessing import chunk_text
except ImportError:
    from phase4_reward import (
        calculate_privacy_leak,
        calculate_reward,
        evaluate_task_gain,
    )
    from phase2_preprocessing import chunk_text



LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

REMOTE_MODEL = os.getenv(
    "REMOTE_MODEL",
    "openai/gpt-oss-120b",
)



_local_pipe = None
_local_generation_config = None
_remote_client = None

def chunk_contains_pii(chunk: str, pii_entities: list[str]) -> bool:
    """Return True when a detected PII unit occurs in a chunk."""
    return contains_pii(chunk, pii_entities)


def build_privacy_mask(chunks: list[str], pii_entities: list[str]) -> torch.Tensor:
    """Boolean mask: True for chunks whose content contains detected PII."""
    return torch.tensor(
        [chunk_contains_pii(chunk, pii_entities) for chunk in chunks],
        dtype=torch.bool,
    )

def calculate_routing_compliance(
    pii_mask: torch.Tensor,
    actual_actions: list[int],
) -> float:
    """
    Measures whether routing follows the desired policy:

        PII chunk     -> LOCAL  (action 0)
        non-PII chunk -> REMOTE (action 1)

    Returns a value in [0, 1].
    """

    if len(actual_actions) == 0:
        return 1.0

    correct = 0

    for is_pii, action in zip(
        pii_mask.tolist(),
        actual_actions
    ):

        if is_pii:
            correct += int(action == 0)

        else:
            correct += int(action == 1)

    return correct / len(actual_actions)


def apply_privacy_constraint(
    action_probs: torch.Tensor,
    pii_mask: torch.Tensor,
) -> torch.Tensor:
    """Force PII chunks to LOCAL while leaving safe chunks unchanged.

    Action 0 = LOCAL, action 1 = REMOTE.
    The returned probabilities remain differentiable for non-PII chunks.
    """
    constrained = action_probs.clone()

    pii_mask = pii_mask.to(device=action_probs.device, dtype=torch.bool)
    pii_mask = pii_mask.unsqueeze(0)

    constrained[..., 0] = torch.where(
        pii_mask,
        torch.ones_like(constrained[..., 0]),
        constrained[..., 0],
    )
    constrained[..., 1] = torch.where(
        pii_mask,
        torch.zeros_like(constrained[..., 1]),
        constrained[..., 1],
    )

    return constrained


def generate_hybrid_response(
    chunks,
    actions,
    pii_entities,
    local_model_fn,
    remote_model_fn,
):

    local_chunks = []
    remote_chunks = []

    actual_actions = []
    pii_forced_local = []

    for chunk, action in zip(chunks, actions):
        chunk_has_pii = chunk_contains_pii(chunk, pii_entities)

        if chunk_has_pii:
            actual_action = 0
            pii_forced = True
        else:
            actual_action = int(action)
            pii_forced = False

        actual_actions.append(actual_action)
        pii_forced_local.append(pii_forced)

        if actual_action == 1:
            remote_chunks.append(chunk)
        else:
            local_chunks.append(chunk)


    local_output = ""

    if local_chunks:
        local_context = " ".join(local_chunks)

        local_output = local_model_fn(local_context)


    remote_output = ""

    if remote_chunks:
        remote_context = " ".join(remote_chunks)

        if contains_pii(remote_context, pii_entities):
            raise RuntimeError("PRIVACY VIOLATION: PII detected in remote payload")

        remote_output = remote_model_fn(remote_context)


    response_parts = [
        output.strip()
        for output in (local_output, remote_output)
        if output and output.strip()
    ]
    final_response = "\n\n".join(response_parts)

    return (
        final_response,
        remote_chunks,
        actual_actions,
        pii_forced_local,
    )

def make_local_llm():
    """Create the cached local LLM generation function."""
    from transformers import GenerationConfig, pipeline

    global _local_pipe, _local_generation_config

    if _local_pipe is None:
        _local_pipe = pipeline(
            "text-generation",
            model=LOCAL_MODEL,
            device_map="auto",
        )

        _local_generation_config = GenerationConfig(
            max_length=None,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=_local_pipe.tokenizer.eos_token_id,
        )


    def _call(text: str) -> str:
        if not text or not text.strip():
            return ""

        output = _local_pipe(
            text,
            generation_config=_local_generation_config,
            clean_up_tokenization_spaces=False,
            return_full_text=False,
        )

        generated_text = output[0]["generated_text"]

        if generated_text.startswith(text):
            generated_text = generated_text[len(text):]

        return generated_text.strip()

    return _call

def make_remote_llm(hf_token, strict=True):

    global _remote_client

    try:
        from .openai_compatible import get_external_client
    except ImportError:
        from openai_compatible import get_external_client

    model = os.getenv("REMOTE_MODEL", REMOTE_MODEL)
    external_client = get_external_client(model)
    if external_client:
        client = external_client
    elif not hf_token:
        raise ValueError(
            "Set LLM_API_KEY and LLM_API_BASE for an external provider, "
            "or provide HF_TOKEN for Hugging Face inference."
        )

    if not external_client and _remote_client is None:

        from huggingface_hub import InferenceClient

        _remote_client = InferenceClient(
            model=model,
            provider="auto",
            token=hf_token,
            timeout=60
        )

    client = client if external_client else _remote_client

    def _call(text):

        if not text or not text.strip():
            return ""

        try:

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=512,
                temperature=0.0
            )

            if not response.choices:
                print("Remote LLM returned no choices.")
                return ""

            message = response.choices[0].message

            if message.content is None:
                print(
                    "Remote LLM returned no final content. "
                    f"Finish reason: "
                    f"{response.choices[0].finish_reason}"
                )
                return ""

            return message.content.strip()

        except Exception as e:

            if "402" in str(e) or "Payment Required" in str(e):
                raise RuntimeError(
                    "Hugging Face remote-model inference credits are exhausted. "
                    "Restore provider credits before end-to-end evaluation."
                ) from e

            if strict:
                raise RuntimeError(
                    f"Remote LLM inference failed: {str(e)[:500]}"
                ) from e
            return ""

    return _call


def execute_real_rollout(
    query,
    pii_entities,
    gold_response,
    agent,
    chunker,
    extractor,
    hf_token,
    debug=True,
    local_llm_fn=None,
    remote_llm_fn=None,
    task_gain_fn=None,
):
    """Execute one privacy-constrained Privacy-R1 rollout."""
    chunks = chunk_text(query, chunker)
    if not chunks:
        chunks = [query]

    embeddings = extractor.get_embeddings(chunks).unsqueeze(0)

    pii_mask = build_privacy_mask(chunks, pii_entities)

    with torch.no_grad():
        action_probs, state_value = agent(embeddings)

        constrained_probs = apply_privacy_constraint(
            action_probs,
            pii_mask,
        )

        distribution = torch.distributions.Categorical(
            probs=constrained_probs
        )
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)

    local_llm = local_llm_fn or make_local_llm()
    remote_llm = remote_llm_fn or make_remote_llm(hf_token, strict=True)

    (
        final_response,
        remote_chunks,
        actual_actions,
        pii_forced_local,
    ) = generate_hybrid_response(
        chunks,
        actions[0].tolist(),
        pii_entities,
        local_llm,
        remote_llm,
    )

    if task_gain_fn:
        task_gain = task_gain_fn(
            redact_pii(query, pii_entities),
            redact_pii(gold_response, pii_entities),
            redact_pii(final_response, pii_entities),
        )
    else:
        task_gain = evaluate_task_gain(
            redact_pii(query, pii_entities),
            redact_pii(gold_response, pii_entities),
            redact_pii(final_response, pii_entities),
            hf_token,
            strict=True,
        )

    privacy_leak = calculate_privacy_leak(
        pii_entities,
        remote_chunks,
    )

    routing_compliance = calculate_routing_compliance(
        pii_mask,
        actual_actions,
    )

    reward = calculate_reward(
        task_gain=task_gain,
        privacy_leak=privacy_leak,
        routing_compliance=routing_compliance,
    )

    trainable_mask = (~pii_mask).unsqueeze(0)

    remote_ratio = (
        len(remote_chunks) / len(chunks)
        if chunks
        else 0.0
    )

    if debug:
        print("\n" + "=" * 70)
        print("ROLLOUT DEBUG")
        print("=" * 70)
        print("QUERY:")
        print(query)

        print("\nPII ENTITIES:")
        print(pii_entities)

        print("\nCHUNKS:")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk}")
            print(f"  Contains PII: {bool(pii_mask[i])}")

        print("\nPPO ACTIONS:")
        for i, action in enumerate(actions[0].tolist()):
            route = "LOCAL" if action == 0 else "REMOTE"
            print(f"Chunk {i} -> {route}")

        print("\nACTUAL ROUTING:")
        for i, action in enumerate(actual_actions):
            route = "LOCAL" if action == 0 else "REMOTE"
            suffix = " [PII FORCED]" if pii_forced_local[i] else ""
            print(f"Chunk {i} -> {route}{suffix}")

        print("\nREMOTE CHUNKS:")
        for chunk in remote_chunks:
            print(chunk)

        print("\nFINAL RESPONSE:")
        print(final_response[:500])

        print("\nGOLD RESPONSE:")
        print(gold_response)

        print("\nROLLOUT METRICS")
        print("-" * 40)
        print(f"Task Gain       : {task_gain:.4f}")
        print(f"Privacy Leak    : {privacy_leak:.4f}")
        print(f"Remote Ratio    : {remote_ratio:.4f}")
        print(f"Routing Compliance: {routing_compliance:.4f}")
        print(f"Reward          : {reward:.4f}")
        print("=" * 70)

    return {
        "embeddings": embeddings,
        "actions": actions,
        "log_prob": log_prob,
        "reward": float(reward),
        "state_value": state_value,
        "privacy_leak": float(privacy_leak),
        "task_gain": float(task_gain),
        "actual_actions": torch.tensor(
            actual_actions,
            dtype=torch.long,
            device=embeddings.device,
        ).unsqueeze(0),
        "trainable_mask": trainable_mask.to(embeddings.device),
        "pii_mask": pii_mask.to(embeddings.device),
        "remote_ratio": float(remote_ratio),
        "routing_compliance": float(routing_compliance),
        "remote_chunks": remote_chunks,
        "final_response": final_response,
    }
