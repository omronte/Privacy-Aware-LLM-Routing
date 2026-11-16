"""Metrics used by routing and ablation evaluation."""

from __future__ import annotations

# pyrefly: ignore[missing-import]
import numpy as np

from src.privacy import detect_pii_spans


def privacy_leakage_rate(pii_entities, remote_chunks):
    remote_text = " ".join(remote_chunks).lower()
    detected_remote_pii = detect_pii_spans(" ".join(remote_chunks), pii_entities)
    if not pii_entities:
        return 1.0 if detected_remote_pii else 0.0

    leaked = sum(
        str(pii).lower() in remote_text
        for pii in pii_entities
    )
    if detected_remote_pii and leaked == 0:
        return 1.0
    return min(1.0, leaked / len(pii_entities))


def remote_routing_ratio(chunks, actions):
    if not chunks:
        return 0.0
    remote_count = sum(int(action) == 1 for action in actions)
    return remote_count / len(chunks)


def average(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def summarize_results(results):
    if not results:
        return {
            "privacy_leakage": 0.0,
            "task_gain": 0.0,
            "reward": 0.0,
            "remote_routing_ratio": 0.0,
            "routing_compliance": 0.0,
            "rollout_failures": 0,
            "successful_rollouts": 0,
        }

    return {
        "privacy_leakage": average(
            [x["privacy_leak"] for x in results]
        ),
        "task_gain": average(
            [x["task_gain"] for x in results]
        ),
        "reward": average(
            [x["reward"] for x in results]
        ),
        "remote_routing_ratio": average(
            [x["remote_routing_ratio"] for x in results]
        ),
        "routing_compliance": average(
            [x["routing_compliance"] for x in results]
        ),
        "rollout_failures": int(sum(x.get("rollout_failed", 0) for x in results)),
        "successful_rollouts": int(
            sum(not x.get("rollout_failed", 0) for x in results)
        ),
    }
