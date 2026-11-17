"""Automated LLM Judge for evaluating generated response quality.

This module provides the LLM-as-a-judge evaluation interface used to score
hybrid local/remote responses against gold reference answers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from src.phase4_reward import (
    JUDGE_MODEL,
    evaluate_task_gain,
    evaluate_task_gain_local,
    get_judge_client,
)


class LLMJudge:
    """Automated LLM Judge for semantic answer evaluation.

    Evaluates generated clinical/task answers against gold reference targets.
    Returns:
        1.0 -> Correct, relevant, and sufficiently complete (Score: 2)
        0.5 -> Partially correct or incomplete (Score: 1)
        0.0 -> Incorrect or irrelevant (Score: 0)
    """

    def __init__(self, model_name: str | None = None, hf_token: str | None = None):
        self.model_name = model_name or os.getenv("JUDGE_MODEL", JUDGE_MODEL)
        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.client = get_judge_client(self.hf_token)

    def evaluate(
        self,
        query: str,
        gold_response: str,
        generated_response: str,
        strict: bool = True,
    ) -> float:
        """Score a generated response against the reference gold answer."""
        return evaluate_task_gain(
            query=query,
            gold_response=gold_response,
            generated_response=generated_response,
            hf_token=self.hf_token,
            strict=strict,
        )

    def evaluate_offline(
        self,
        query: str,
        gold_response: str,
        generated_response: str,
    ) -> float:
        """Fast offline semantic similarity judge using MiniLM embeddings."""
        return evaluate_task_gain_local(
            query=query,
            gold_response=gold_response,
            generated_response=generated_response,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What medication is prescribed for hypertension?",
        help="Input query",
    )
    parser.add_argument(
        "--gold",
        default="ACE inhibitors or ARBs such as lisinopril or losartan.",
        help="Gold reference answer",
    )
    parser.add_argument(
        "--pred",
        default="Lisinopril is commonly used to treat high blood pressure.",
        help="Generated candidate answer",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline MiniLM cosine similarity judge instead of remote API",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AUTOMATED LLM JUDGE EVALUATION")
    print("=" * 60)
    print(f"Query     : {args.query}")
    print(f"Reference : {args.gold}")
    print(f"Candidate : {args.pred}")
    print("-" * 60)

    judge = LLMJudge()
    if args.offline:
        score = judge.evaluate_offline(args.query, args.gold, args.pred)
        print(f"Judge Type: Offline MiniLM Cosine Similarity")
    else:
        print(f"Judge Model: {judge.model_name}")
        score = judge.evaluate(args.query, args.gold, args.pred)

    print(f"Task Gain Score: {score} (1.0 = Correct, 0.5 = Partial, 0.0 = Incorrect)")
    print("=" * 60)
    return score


if __name__ == "__main__":
    main()
