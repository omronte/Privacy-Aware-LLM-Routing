"""Validate a local Privacy-R1 JSON dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "query",
    "pii_units",
    "target_response",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Dataset must contain a JSON list.")
    return data


def validate_dataset(dataset):
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"\nTotal samples: {len(dataset)}")

    missing_fields = 0
    empty_queries = 0
    empty_pii = 0
    empty_targets = 0
    invalid_chunks = 0
    pii_counts = []

    for index, sample in enumerate(dataset):
        missing = REQUIRED_FIELDS - set(sample.keys())
        if missing:
            missing_fields += 1
            print(f"Sample {index}: missing {sorted(missing)}")

        query = str(sample.get("query", "")).strip()
        target = str(sample.get("target_response", "")).strip()
        pii = sample.get("pii_units", [])
        chunks = sample.get("chunks")

        if not query:
            empty_queries += 1
        if not target:
            empty_targets += 1
        if not pii:
            empty_pii += 1
        if chunks is not None and not isinstance(chunks, list):
            invalid_chunks += 1

        pii_counts.append(len(pii) if isinstance(pii, list) else 0)

    average_pii = (
        sum(pii_counts) / len(pii_counts)
        if pii_counts
        else 0.0
    )

    print("\nValidation Results")
    print("-" * 40)
    print(f"Missing fields    : {missing_fields}")
    print(f"Empty queries     : {empty_queries}")
    print(f"Empty PII         : {empty_pii}")
    print(f"Empty targets     : {empty_targets}")
    print(f"Invalid chunks    : {invalid_chunks}")
    print(f"Average PII/sample: {average_pii:.2f}")

    passed = (
        missing_fields == 0
        and empty_queries == 0
        and empty_targets == 0
        and invalid_chunks == 0
    )

    print("\nSTATUS:", "PASS" if passed else "FAIL")
    print("=" * 60)

    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    passed = validate_dataset(dataset)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
