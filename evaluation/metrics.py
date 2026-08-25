"""Metrics used by routing and ablation evaluation."""

from __future__ import annotations

import numpy as np


def privacy_leakage_rate(pii_entities, remote_chunks):
    if not pii_entities:
        return 0.0

    remote_text = " ".join(remote_chunks).lower()
    leaked = sum(
        str(pii).lower() in remote_text
        for pii in pii_entities
    )
    return leaked / len(pii_entities)


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
    }
