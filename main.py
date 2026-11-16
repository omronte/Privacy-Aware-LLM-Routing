from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.phase2_preprocessing import (
    MiniLMFeatureExtractor,
    chunk_text,
    initialize_chunker,
)

from src.phase3_agent import PrivacyR1Agent

from src.phase6_warmup import (
    generate_heuristic_labels,
    train_supervised_warmup,
)

from src.phase7_ppo import run_ppo_training



ROOT_DIR = Path(__file__).resolve().parent




LOCAL_ACTION = 0
REMOTE_ACTION = 1

ROUTING_POLICY_NAME = (
    "PII -> LOCAL | non-PII -> REMOTE"
)



def load_json_dataset(path: Path):

    with path.open(
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    if not isinstance(
        dataset,
        list
    ) or not dataset:

        raise ValueError(
            f"Dataset must be a non-empty JSON list: {path}"
        )

    required = {
        "query",
        "pii_units",
        "target_response",
    }

    for index, sample in enumerate(dataset):

        missing = (
            required
            - set(sample.keys())
        )

        if missing:

            raise ValueError(
                f"{path}: record {index} is missing "
                f"{sorted(missing)}. "
                "PPO training requires query, pii_units "
                "and target_response."
            )

        if not isinstance(
            sample["pii_units"],
            list
        ):

            raise ValueError(
                f"{path}: record {index} has invalid "
                "'pii_units'. Expected a list."
            )

        if not str(
            sample["query"]
        ).strip():

            raise ValueError(
                f"{path}: record {index} has an empty query."
            )

        if not str(
            sample["target_response"]
        ).strip():

            raise ValueError(
                f"{path}: record {index} has an empty "
                "target_response."
            )

    return dataset



def print_dataset_statistics(dataset):

    records_with_pii = sum(
        bool(sample.get("pii_units"))
        for sample in dataset
    )

    records_without_pii = (
        len(dataset)
        - records_with_pii
    )

    print()
    print("=" * 70)
    print("DATASET ROUTING STATISTICS")
    print("=" * 70)

    print(
        f"Total records       : {len(dataset)}"
    )

    print(
        f"Records with PII    : {records_with_pii}"
    )

    print(
        f"Records without PII : {records_without_pii}"
    )

    if len(dataset) > 0:

        pii_percentage = (
            records_with_pii
            / len(dataset)
            * 100
        )

        print(
            f"PII record ratio    : {pii_percentage:.2f}%"
        )

    print(
        f"Routing policy      : {ROUTING_POLICY_NAME}"
    )

    print("=" * 70)

    return (
        records_with_pii,
        records_without_pii,
    )



def validate_routing_dataset(dataset):

    records_with_pii = sum(
        bool(sample.get("pii_units"))
        for sample in dataset
    )

    if records_with_pii == 0:

        raise ValueError(
            "\n"
            + "=" * 70
            + "\n"
            + "DATASET ERROR: NO PII LABELS FOUND"
            + "\n"
            + "=" * 70
            + "\n"
            + "\n"
            + "Every training record currently has:"
            + "\n"
            + "    pii_units = []"
            + "\n"
            + "Therefore supervised warm-up cannot learn:"
            + "\n"
            + "    PII     -> LOCAL"
            + "\n"
            + "    non-PII -> REMOTE"
            + "\n"
            + "\n"
            + "Do NOT run another 300-step PPO training run "
            + "until the training data contains PII labels."
            + "\n"
            + "\n"
            + "The rollout detector may still detect PII at "
            + "runtime, but the supervised warm-up is currently "
            + "being trained with no PII labels."
            + "\n"
            + "=" * 70
        )



def build_warmup_dataloader(
    dataset,
    chunker,
    extractor,
):

    """
    Create the list-based warm-up loader expected by
    Phase 6.

    Desired policy:

        PII chunk     -> LOCAL  (action 0)
        non-PII chunk -> REMOTE (action 1)
    """

    dataloader = []

    for sample in dataset:

        chunks = chunk_text(
            sample["query"],
            chunker
        )

        if not chunks:

            chunks = [
                sample["query"]
            ]

        embeddings = (
            extractor
            .get_embeddings(chunks)
            .unsqueeze(0)
        )

        labels = generate_heuristic_labels(
            chunks,
            sample.get(
                "pii_units",
                []
            ),
        ).unsqueeze(0)

        dataloader.append(
            (
                embeddings,
                labels
            )
        )

    return dataloader



def print_routing_policy():

    print()
    print("=" * 70)
    print("PRIVACY-AWARE ROUTING POLICY")
    print("=" * 70)

    print(
        "PII chunk     -> LOCAL"
    )

    print(
        "non-PII chunk -> REMOTE"
    )

    print()
    print(
        "Action 0 = LOCAL"
    )

    print(
        "Action 1 = REMOTE"
    )

    print()
    print(
        "Routing compliance = correctly routed chunks"
        " / total chunks"
    )

    print("=" * 70)



def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )


    parser.add_argument(
        "--train-file",
        type=Path,
        default=(
            ROOT_DIR
            / "data"
            / "train.json"
        ),
    )


    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help=(
            "Number of PPO rollouts. "
            "Use 1-2 only for a smoke test. "
            "Use 300+ for actual training."
        ),
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for warm-up, PPO sampling, and dataset ordering.",
    )


    parser.add_argument(
        "--skip-warmup",
        action="store_true",
    )

    parser.add_argument(
        "--no-debug-rollout",
        action="store_true",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    if args.steps < 1:

        raise ValueError(
            "--steps must be at least 1."
        )

    if args.warmup_epochs < 0:

        raise ValueError(
            "--warmup-epochs cannot be negative."
        )

    if args.ppo_batch_size < 1:

        raise ValueError(
            "--ppo-batch-size must be at least 1."
        )

    if args.ppo_epochs < 1:

        raise ValueError(
            "--ppo-epochs must be at least 1."
        )


    load_dotenv(
        ROOT_DIR / ".env"
    )

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )
    )

    if not hf_token:

        raise RuntimeError(
            "Hugging Face token not found. "
            "Set HF_TOKEN in .env before running "
            "real PPO training."
        )


    train_file = args.train_file

    if not train_file.is_absolute():

        train_file = (
            ROOT_DIR
            / train_file
        )

    if not train_file.exists():

        raise FileNotFoundError(
            f"Training file not found:\n{train_file}"
        )


    dataset = load_json_dataset(
        train_file
    )


    print_dataset_statistics(
        dataset
    )


    if not args.skip_warmup:

        validate_routing_dataset(
            dataset
        )


    print_routing_policy()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "PRIVACY-R1 TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Training file   : {train_file}"
    )

    print(
        f"Training records: {len(dataset)}"
    )

    print(
        f"PPO steps       : {args.steps}"
    )

    print(
        f"Warm-up epochs  : {args.warmup_epochs}"
    )

    print(
        f"PPO batch size  : {args.ppo_batch_size}"
    )

    print(
        f"PPO epochs      : {args.ppo_epochs}"
    )

    print(
        f"Routing policy  : {ROUTING_POLICY_NAME}"
    )

    print(
        "=" * 70
    )


    chunker = initialize_chunker()

    extractor = MiniLMFeatureExtractor(
        device=device
    )


    agent = PrivacyR1Agent(
        embedding_dim=extractor.embedding_dim,
    ).to(device)


    if not args.skip_warmup:

        print(
            "\n[1/3] Running supervised privacy warm-up..."
        )

        warmup_loader = (
            build_warmup_dataloader(
                dataset,
                chunker,
                extractor,
            )
        )

        train_supervised_warmup(
            agent=agent,
            dataloader=warmup_loader,
            epochs=args.warmup_epochs,
        )

    else:

        print(
            "\n[1/3] Supervised warm-up skipped."
        )


    print(
        "\n[2/3] Running PPO training..."
    )

    print()
    print(
        "Expected routing:"
    )

    print(
        "  PII     -> LOCAL"
    )

    print(
        "  non-PII -> REMOTE"
    )

    print()


    results = run_ppo_training(
        agent=agent,
        chunker=chunker,
        extractor=extractor,
        hf_token=hf_token,
        dataset=dataset,
        num_steps=args.steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_epochs=args.ppo_epochs,
        results_dir=(
            ROOT_DIR
            / "results"
        ),
        debug_first_rollout=(
            not args.no_debug_rollout
        ),

        routing_policy=ROUTING_POLICY_NAME,
        local_action=LOCAL_ACTION,
        remote_action=REMOTE_ACTION,
        seed=args.seed,
    )


    print(
        "\n[3/3] Training complete."
    )

    print(
        f"Training CSV : {results['training_csv']}"
    )

    print(
        f"Summary JSON : {results['summary_json']}"
    )

    print(
        f"Model        : {results['model_path']}"
    )

    print(
        "=" * 70
    )



if __name__ == "__main__":

    main()
