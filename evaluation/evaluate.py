"""Fast routing-policy evaluation without LLM generation.

This script measures privacy leakage, routing compliance, and remote-routing
ratio. It does not claim a task-quality score because no model response is
generated here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.baselines import (
    all_local_actions,
    all_remote_actions,
    heuristic_actions,
)
from evaluation.metrics import (
    privacy_leakage_rate,
    remote_routing_ratio,
    summarize_results,
)
from src.phase2_preprocessing import (
    MiniLMFeatureExtractor,
    chunk_text,
    initialize_chunker,
)
from src.phase3_agent import PrivacyR1Agent
from src.phase4_reward import evaluate_task_gain_local
from src.phase5_rollout import (
    apply_privacy_constraint,
    build_privacy_mask,
    calculate_routing_compliance,
    execute_real_rollout,
    make_local_llm,
)


def _production_chunks(sample, chunker):
    chunks = chunk_text(sample["query"], chunker)
    return chunks or [sample["query"]]


def evaluate_policy(dataset, policy, chunker):
    results = []

    for sample in dataset:
        chunks = _production_chunks(sample, chunker)
        pii_entities = sample.get("pii_units", [])
        actions = policy(chunks, pii_entities)

        remote_chunks = [
            chunk
            for chunk, action in zip(chunks, actions)
            if action == 1
        ]

        results.append(
            {
                "privacy_leak": privacy_leakage_rate(
                    pii_entities,
                    remote_chunks,
                ),
                "task_gain": 0.0,
                "reward": 0.0,
                "remote_routing_ratio": remote_routing_ratio(
                    chunks,
                    actions,
                ),
                "routing_compliance": calculate_routing_compliance(
                    build_privacy_mask(chunks, pii_entities),
                    actions,
                ),
            }
        )

    return summarize_results(results)


def evaluate_dataset(dataset, checkpoint=None, device=None):
    chunker = initialize_chunker()
    policies = {
        "all_local": lambda chunks, _: all_local_actions(chunks),
        "all_remote": lambda chunks, _: all_remote_actions(chunks),
        "heuristic": heuristic_actions,
    }

    results = {
        name: evaluate_policy(dataset, policy, chunker)
        for name, policy in policies.items()
    }

    if checkpoint:
        results["trained_checkpoint"] = evaluate_checkpoint(
            dataset,
            checkpoint,
            device=device,
            chunker=chunker,
        )

    return results


def evaluate_checkpoint_end_to_end(
    dataset,
    checkpoint_path,
    hf_token,
    device=None,
    chunker=None,
    max_samples=None,
    offline=False,
):
    """Evaluate routing and generated-answer quality with real model calls."""
    has_remote = bool(
        hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("LLM_API_KEY")
    )
    if not has_remote and not offline:
        raise ValueError(
            "HF_TOKEN or LLM_API_KEY is required for --with-task-gain evaluation."
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
        candidate = ROOT_DIR / checkpoint_path
        if candidate.exists():
            checkpoint_path = candidate
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    chunker = chunker or initialize_chunker()
    extractor = MiniLMFeatureExtractor(device=device)
    agent = PrivacyR1Agent(embedding_dim=extractor.embedding_dim).to(device)
    agent.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    agent.eval()
    local_llm = make_local_llm() if offline else None

    results = []
    failure_reasons = Counter()
    samples = dataset[:max_samples] if max_samples else dataset
    for sample in samples:
        try:
            rollout = execute_real_rollout(
                query=sample["query"],
                pii_entities=sample.get("pii_units", []),
                gold_response=sample["target_response"],
                agent=agent,
                chunker=chunker,
                extractor=extractor,
                hf_token=hf_token,
                debug=False,
                local_llm_fn=local_llm,
                remote_llm_fn=local_llm,
                task_gain_fn=(
                    evaluate_task_gain_local
                    if offline
                    else None
                ),
            )
            results.append(
                {
                    "privacy_leak": rollout["privacy_leak"],
                    "task_gain": rollout["task_gain"],
                    "reward": rollout["reward"],
                    "remote_routing_ratio": rollout["remote_ratio"],
                    "routing_compliance": rollout["routing_compliance"],
                    "rollout_failed": 0,
                }
            )
        except Exception as exc:
            message = str(exc)
            if hf_token and isinstance(hf_token, str):
                message = message.replace(hf_token, "<redacted-token>")
            failure_reasons[
                f"{type(exc).__name__}: {message[:240]}"
            ] += 1
            results.append(
                {
                    "privacy_leak": 0.0,
                    "task_gain": 0.0,
                    "reward": 0.0,
                    "remote_routing_ratio": 0.0,
                    "routing_compliance": 0.0,
                    "rollout_failed": 1,
                }
            )

    summary = summarize_results(results)
    summary["failure_reasons"] = dict(failure_reasons)
    return summary


def evaluate_checkpoint(dataset, checkpoint_path, device=None, chunker=None):
    """Evaluate a saved agent deterministically on the production pipeline."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
        candidate = ROOT_DIR / checkpoint_path
        if candidate.exists():
            checkpoint_path = candidate
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    chunker = chunker or initialize_chunker()
    extractor = MiniLMFeatureExtractor(device=device)
    agent = PrivacyR1Agent(embedding_dim=extractor.embedding_dim).to(device)
    agent.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    agent.eval()

    results = []
    with torch.no_grad():
        for sample in dataset:
            chunks = _production_chunks(sample, chunker)
            pii_entities = sample.get("pii_units", [])
            embeddings = extractor.get_embeddings(chunks).unsqueeze(0)
            pii_mask = build_privacy_mask(chunks, pii_entities)
            action_probs, _ = agent(embeddings)
            constrained_probs = apply_privacy_constraint(
                action_probs,
                pii_mask,
            )
            actions = constrained_probs.argmax(dim=-1)[0].tolist()
            remote_chunks = [
                chunk
                for chunk, action in zip(chunks, actions)
                if action == 1
            ]
            results.append(
                {
                    "privacy_leak": privacy_leakage_rate(
                        pii_entities,
                        remote_chunks,
                    ),
                    "task_gain": 0.0,
                    "reward": 0.0,
                    "remote_routing_ratio": remote_routing_ratio(
                        chunks,
                        actions,
                    ),
                    "routing_compliance": calculate_routing_compliance(
                        pii_mask,
                        actions,
                    ),
                }
            )

    return summarize_results(results)


def load_dataset(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        candidate = ROOT_DIR / path
        if candidate.exists():
            path = candidate
    with path.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    if not isinstance(dataset, list):
        raise ValueError("Dataset JSON must contain a list of records.")

    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset = ROOT_DIR / "data" / "test.json"
    default_checkpoint = (
        ROOT_DIR / "models" / "privacy_r1_agent.pt"
        if (ROOT_DIR / "models" / "privacy_r1_agent.pt").exists()
        else None
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=default_dataset,
        help="Path to dataset JSON (defaults to data/test.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Optional saved PrivacyR1Agent checkpoint to evaluate (defaults to models/privacy_r1_agent.pt if found).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Device for checkpoint evaluation; defaults to CUDA when available.",
    )
    parser.add_argument(
        "--with-task-gain",
        action="store_true",
        help="Run local/remote generation and remote judge scoring.",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token; defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit end-to-end evaluation samples for debugging.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index to evaluate.",
    )
    parser.add_argument(
        "--offline-task-gain",
        action="store_true",
        help="Use local generation and MiniLM similarity instead of HF inference.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.start_index < 0:
        parser.error("--start-index cannot be negative")
    end_index = (
        args.start_index + args.max_samples
        if args.max_samples is not None
        else len(dataset)
    )
    dataset = dataset[args.start_index:end_index]
    if not dataset:
        parser.error("The selected dataset slice is empty")
    results = evaluate_dataset(
        dataset,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    if args.with_task_gain:
        if not args.checkpoint:
            parser.error("--with-task-gain requires --checkpoint")
        load_dotenv(ROOT_DIR / ".env")
        results["trained_checkpoint_end_to_end"] = (
            evaluate_checkpoint_end_to_end(
                dataset,
                args.checkpoint,
                args.hf_token
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_TOKEN")
                or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                device=args.device,
                max_samples=None,
                offline=args.offline_task_gain,
            )
        )
    serialized = json.dumps(results, indent=2)
    print(serialized)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    return results


if __name__ == "__main__":
    main()
