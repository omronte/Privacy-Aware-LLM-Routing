"""Fast routing-policy evaluation without LLM generation.

This script measures privacy leakage and remote-routing ratio only. It does not
claim a task-quality score because no model response is generated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
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


def evaluate_policy(dataset, policy):
    results = []

    for sample in dataset:
        chunks = sample.get("chunks") or [sample.get("query", "")]
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
            }
        )

    return summarize_results(results)


def evaluate_dataset(dataset):
    policies = {
        "all_local": lambda chunks, _: all_local_actions(chunks),
        "all_remote": lambda chunks, _: all_remote_actions(chunks),
        "heuristic": heuristic_actions,
    }

    return {
        name: evaluate_policy(dataset, policy)
        for name, policy in policies.items()
    }


def load_dataset(path):
    with Path(path).open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    if not isinstance(dataset, list):
        raise ValueError("Dataset JSON must contain a list of records.")

    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = evaluate_dataset(load_dataset(args.dataset))
    serialized = json.dumps(results, indent=2)
    print(serialized)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    return results


if __name__ == "__main__":
    main()
