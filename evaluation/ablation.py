"""Ablation analysis for the built-in routing policies."""

import argparse
import json
from pathlib import Path

from evaluation.evaluate import evaluate_dataset, load_dataset


def compare_policies(dataset):
    """Return policy metrics plus privacy deltas against all-remote routing."""
    results = evaluate_dataset(dataset)
    remote_privacy = results["all_remote"]["privacy_leakage"]

    for policy_result in results.values():
        policy_result["privacy_reduction_vs_all_remote"] = (
            remote_privacy - policy_result["privacy_leakage"]
        )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a JSON dataset")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving the ablation JSON results"
    )
    args = parser.parse_args()

    results = compare_policies(load_dataset(args.dataset))
    serialized = json.dumps(results, indent=2)
    print(serialized)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    return results


if __name__ == "__main__":
    main()
