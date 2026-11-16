"""Merge disjoint evaluation summary files into one aggregate report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


AVERAGE_FIELDS = (
    "privacy_leakage",
    "task_gain",
    "reward",
    "remote_routing_ratio",
    "routing_compliance",
)


def merge_reports(paths):
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    names = set().union(*(report.keys() for report in reports))
    merged = {}

    for name in sorted(names):
        records = [report[name] for report in reports if name in report]
        counts = [
            record.get("successful_rollouts", record.get("sample_count", 0))
            + record.get("rollout_failures", 0)
            for record in records
        ]
        total = sum(counts) or len(records)
        result = {}
        for field in AVERAGE_FIELDS:
            result[field] = sum(
                record.get(field, 0.0) * count
                for record, count in zip(records, counts)
            ) / total
        result["rollout_failures"] = sum(
            record.get("rollout_failures", 0) for record in records
        )
        result["successful_rollouts"] = sum(
            record.get("successful_rollouts", 0) for record in records
        )
        failure_reasons = Counter()
        for record in records:
            failure_reasons.update(record.get("failure_reasons", {}))
        if failure_reasons:
            result["failure_reasons"] = dict(failure_reasons)
        if "privacy_reduction_vs_all_remote" in records[0]:
            result["privacy_reduction_vs_all_remote"] = (
                merged.get("all_remote", {}).get("privacy_leakage", 0.0)
                - result["privacy_leakage"]
            )
        merged[name] = result

    if "all_remote" in merged:
        remote_leakage = merged["all_remote"]["privacy_leakage"]
        for result in merged.values():
            if "privacy_leakage" in result:
                result["privacy_reduction_vs_all_remote"] = (
                    remote_leakage - result["privacy_leakage"]
                )
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge_reports(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()