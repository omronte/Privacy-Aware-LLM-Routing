"""Create PPO training curves from results/raw/ppo_training.csv."""

from __future__ import annotations

import csv
from pathlib import Path

# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore[missing-import]
import numpy as np



ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT_DIR
    / "results"
    / "raw"
    / "ppo_training.csv"
)

DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "results"
    / "figures"
)



def load_history(input_path=DEFAULT_INPUT):

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training results file not found:\n{input_path}"
        )

    with input_path.open(
        newline="",
        encoding="utf-8"
    ) as csv_file:

        rows = list(
            csv.DictReader(csv_file)
        )

    if not rows:
        raise ValueError(
            f"Training results file is empty:\n{input_path}"
        )


    required_columns = {
        "step",
        "reward",
        "privacy_leak",
        "task_gain",
        "remote_ratio",
        "routing_compliance",
    }

    actual_columns = set(rows[0].keys())

    missing_columns = (
        required_columns - actual_columns
    )

    if missing_columns:

        raise ValueError(
            "\nThe training CSV is missing required "
            "columns:\n\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing_columns)
            )
            + "\n\n"
            "Your PPO training code must save "
            "'routing_compliance' to the CSV before "
            "this graph can be generated."
        )


    return {

        "step": np.array(
            [
                float(row["step"])
                for row in rows
            ]
        ),

        "reward": np.array(
            [
                float(row["reward"])
                for row in rows
            ]
        ),

        "privacy_leak": np.array(
            [
                float(row["privacy_leak"])
                for row in rows
            ]
        ),

        "task_gain": np.array(
            [
                float(row["task_gain"])
                for row in rows
            ]
        ),

        "remote_ratio": np.array(
            [
                float(row["remote_ratio"])
                for row in rows
            ]
        ),

        "routing_compliance": np.array(
            [
                float(row["routing_compliance"])
                for row in rows
            ]
        ),
    }



def moving_average(
    values,
    window=10
):

    if len(values) < 2:
        return values

    window = min(
        window,
        len(values)
    )

    kernel = (
        np.ones(window) / window
    )

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )



def _save_curve(
    x,
    y,
    title,
    ylabel,
    output_path,
    y_limits=None,
):

    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(
        x,
        y,
        label="Episode"
    )


    if len(y) >= 2:

        ma = moving_average(
            y,
            window=10
        )

        ma_x = x[
            len(x) - len(ma):
        ]

        plt.plot(
            ma_x,
            ma,
            label="10-episode moving average"
        )


    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )


    if y_limits is not None:

        plt.ylim(
            y_limits[0],
            y_limits[1]
        )


    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_all(
    input_path=DEFAULT_INPUT,
    output_dir=DEFAULT_OUTPUT_DIR
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history = load_history(
        input_path
    )

    x = history["step"]

    outputs = {}


    output = (
        output_dir
        / "reward_curve.png"
    )

    _save_curve(
        x,
        history["reward"],
        "Reward During PPO Training",
        "Reward",
        output,
    )

    outputs["reward"] = output


    output = (
        output_dir
        / "privacy_leak_curve.png"
    )

    _save_curve(
        x,
        history["privacy_leak"],
        "Privacy Leakage During PPO Training",
        "Privacy Leak",
        output,
        y_limits=(0, 1.05),
    )

    outputs["privacy_leak"] = output


    output = (
        output_dir
        / "task_gain_curve.png"
    )

    _save_curve(
        x,
        history["task_gain"],
        "Task Gain During PPO Training",
        "Task Gain",
        output,
        y_limits=(0, 1.05),
    )

    outputs["task_gain"] = output


    output = (
        output_dir
        / "remote_ratio_curve.png"
    )

    _save_curve(
        x,
        history["remote_ratio"],
        "Remote Routing Ratio During PPO Training",
        "Remote Routing Ratio",
        output,
        y_limits=(0, 1.05),
    )

    outputs["remote_ratio"] = output


    output = (
        output_dir
        / "routing_compliance_curve.png"
    )

    _save_curve(
        x,
        history["routing_compliance"],
        "Routing Compliance During PPO Training",
        "Routing Compliance",
        output,
        y_limits=(0, 1.05),
    )

    outputs["routing_compliance"] = output

    return outputs



def plot_reward_curve(
    input_path=DEFAULT_INPUT,
    output_path=None
):

    if output_path is None:

        output_path = (
            DEFAULT_OUTPUT_DIR
            / "reward_curve.png"
        )

    history = load_history(
        input_path
    )

    _save_curve(
        history["step"],
        history["reward"],
        "Reward During PPO Training",
        "Reward",
        Path(output_path),
    )

    return Path(
        output_path
    )



def plot_routing_compliance(
    input_path=DEFAULT_INPUT,
    output_path=None
):

    if output_path is None:

        output_path = (
            DEFAULT_OUTPUT_DIR
            / "routing_compliance_curve.png"
        )

    history = load_history(
        input_path
    )

    _save_curve(
        history["step"],
        history["routing_compliance"],
        "Routing Compliance During PPO Training",
        "Routing Compliance",
        Path(output_path),
        y_limits=(0, 1.05),
    )

    return Path(
        output_path
    )



if __name__ == "__main__":

    print("=" * 60)
    print("GENERATING PPO TRAINING FIGURES")
    print("=" * 60)

    try:

        generated = plot_all()

        print()

        for name, path in generated.items():

            print(
                f"Saved {name} curve to:"
            )

            print(
                f"  {path}"
            )

        print()
        print("=" * 60)
        print("ALL FIGURES GENERATED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("FIGURE GENERATION FAILED")
        print("=" * 60)

        print(
            str(e)
        )

        raise
