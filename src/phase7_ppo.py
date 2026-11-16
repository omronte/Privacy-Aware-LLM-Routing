
from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm.auto import tqdm

try:
    from .phase5_rollout import execute_real_rollout
except ImportError:
    from phase5_rollout import execute_real_rollout


class PPOTrainer:
    """PPO trainer for sequence-level routing decisions.

    Each query produces one terminal reward. The same episode reward is used
    for every learnable (non-PII) chunk through A_t = R - V(s_t).
    PII chunks are hard-constrained to LOCAL and excluded from the policy loss.
    """

    def __init__(
        self,
        agent,
        lr=1e-5,
        clip_param=0.2,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=8,
        ppo_epochs=4,
    ):
        self.agent = agent
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=lr,
        )
        self.clip_param = clip_param
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.memory = []

    def store_transition(
        self,
        state,
        action,
        log_prob,
        reward,
        value,
        trainable_mask,
    ):
        self.memory.append(
            {
                "state": state.detach(),
                "action": action.detach(),
                "log_prob": log_prob.detach(),
                "reward": float(reward),
                "value": value.detach(),
                "trainable_mask": trainable_mask.detach(),
            }
        )

    @staticmethod
    def _flatten_episode(tensor):
        """Flatten a [1, chunks, ...] episode tensor into chunk entries."""
        return tensor.squeeze(0)

    def _calculate_advantages(self):
        """Calculate terminal-reward advantages for every stored chunk."""
        all_advantages = []
        all_returns = []

        for transition in self.memory:
            value = self._flatten_episode(
                transition["value"].squeeze(-1)
            )
            reward = torch.full_like(
                value,
                float(transition["reward"]),
            )

            advantages = reward - value
            returns = reward

            all_advantages.append(advantages.detach())
            all_returns.append(returns.detach())

        learnable = []
        for transition, advantages in zip(
            self.memory,
            all_advantages,
        ):
            mask = self._flatten_episode(
                transition["trainable_mask"]
            ).bool()
            if mask.any():
                learnable.append(advantages[mask])

        if learnable:
            joined = torch.cat(learnable)
            mean = joined.mean()
            std = joined.std(unbiased=False).clamp_min(1e-8)

            normalized = []
            for transition, advantages in zip(
                self.memory,
                all_advantages,
            ):
                normalized.append((advantages - mean) / std)
            all_advantages = normalized

        return all_advantages, all_returns

    def update(self):
        if not self.memory:
            return None

        advantages_list, returns_list = self._calculate_advantages()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        self.agent.train()

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()

            batch_loss = None
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0

            for idx, transition in enumerate(self.memory):
                state = transition["state"]
                action = transition["action"]
                old_log_prob = transition["log_prob"]
                trainable_mask = transition["trainable_mask"].bool()

                new_probs, new_values = self.agent(state)
                new_probs = new_probs.clamp(
                    min=1e-8,
                    max=1.0 - 1e-8,
                )

                distribution = Categorical(probs=new_probs)
                new_log_prob = distribution.log_prob(action)
                entropy = distribution.entropy()

                mask = trainable_mask.expand_as(new_log_prob)

                if mask.any():
                    ratio = torch.exp(
                        new_log_prob[mask] - old_log_prob[mask]
                    )

                    advantage = advantages_list[idx].to(
                        new_log_prob.device
                    )[mask.squeeze(0)]

                    surrogate_1 = ratio * advantage
                    surrogate_2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    ) * advantage

                    policy_loss = -torch.min(
                        surrogate_1,
                        surrogate_2,
                    ).mean()
                    entropy_loss = entropy[mask].mean()
                else:
                    policy_loss = torch.zeros(
                        (),
                        device=new_probs.device,
                    )
                    entropy_loss = torch.zeros(
                        (),
                        device=new_probs.device,
                    )

                predicted_values = new_values.squeeze(0).squeeze(-1)
                target_returns = returns_list[idx].to(
                    predicted_values.device
                )

                value_loss = nn.functional.mse_loss(
                    predicted_values,
                    target_returns,
                )

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                batch_loss = (
                    loss
                    if batch_loss is None
                    else batch_loss + loss
                )

                epoch_policy_loss += float(policy_loss.item())
                epoch_value_loss += float(value_loss.item())
                epoch_entropy += float(entropy_loss.item())
                update_count += 1

            if batch_loss is not None:
                batch_loss = batch_loss / max(len(self.memory), 1)
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()

            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy += epoch_entropy

        num_epochs = max(self.ppo_epochs, 1)
        denominator = max(len(self.memory) * num_epochs, 1)

        average_loss = (
            total_policy_loss
            + self.value_coef * total_value_loss
            - self.entropy_coef * total_entropy
        ) / denominator

        self.memory.clear()

        return float(average_loss)


def _validate_training_record(sample, index):
    required = ("query", "pii_units", "target_response")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(
            f"Training record {index} is missing required field(s): "
            f"{', '.join(missing)}. "
            "PPO reward training requires target_response."
        )


def run_ppo_training(
    agent,
    chunker,
    extractor,
    hf_token,
    dataset,
    num_steps=100,
    ppo_batch_size=8,
    ppo_epochs=4,
    results_dir="results",
    debug_first_rollout=False,
    routing_policy=None,
    local_action=0,
    remote_action=1,
    seed=42,
):
    """Run PPO over the supplied training set and save reproducible outputs."""
    if not dataset:
        raise ValueError("Training dataset is empty.")

    for index, sample in enumerate(dataset):
        _validate_training_record(sample, index)

    trainer = PPOTrainer(
        agent=agent,
        lr=1e-5,
        clip_param=0.2,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=ppo_batch_size,
        ppo_epochs=ppo_epochs,
    )

    reward_history = []
    privacy_history = []
    task_gain_history = []
    remote_ratio_history = []
    loss_history = []
    step_history = []
    routing_compliance_history = []

    reward_window = deque(maxlen=25)
    privacy_window = deque(maxlen=25)
    task_window = deque(maxlen=25)
    remote_window = deque(maxlen=25)
    routing_window = deque(maxlen=25)

    print(
        f"Starting PPO training for {num_steps} steps "
        f"using {len(dataset)} training records..."
    )

    rng = np.random.default_rng(seed)
    training_indices = rng.permutation(len(dataset))

    for step in tqdm(range(num_steps), desc="PPO Training"):
        sample = dataset[int(training_indices[step % len(training_indices)])]

        query = sample["query"]
        pii_entities = sample.get("pii_units", [])
        gold_response = sample["target_response"]

        try:
            rollout = execute_real_rollout(
                query=query,
                pii_entities=pii_entities,
                gold_response=gold_response,
                agent=agent,
                chunker=chunker,
                extractor=extractor,
                hf_token=hf_token,
                debug=debug_first_rollout and step == 0,
            )
        except Exception as exc:
            print(f"Rollout failed at step {step + 1}: {exc}")
            raise

        trainer.store_transition(
            rollout["embeddings"],
            rollout["actions"],
            rollout["log_prob"],
            rollout["reward"],
            rollout["state_value"],
            rollout["trainable_mask"],
        )

        reward = rollout["reward"]
        privacy_leak = rollout["privacy_leak"]
        task_gain = rollout["task_gain"]
        remote_ratio = rollout["remote_ratio"]
        routing_compliance = rollout["routing_compliance"]

        reward_history.append(reward)
        privacy_history.append(privacy_leak)
        task_gain_history.append(task_gain)
        remote_ratio_history.append(remote_ratio)
        routing_compliance_history.append(routing_compliance)
        step_history.append(step + 1)

        reward_window.append(reward)
        privacy_window.append(privacy_leak)
        task_window.append(task_gain)
        remote_window.append(remote_ratio)
        routing_window.append(routing_compliance)

        if len(trainer.memory) >= trainer.batch_size:
            loss = trainer.update()
            if loss is not None:
                loss_history.append(loss)

        if (step + 1) % 25 == 0 or step == 0:
            print("\n" + "=" * 60)
            print(f"Step {step + 1}/{num_steps}")
            print(f"Average Reward       : {np.mean(reward_window):.4f}")
            print(
                f"Average Privacy Leak : "
                f"{100 * np.mean(privacy_window):.2f}%"
            )
            print(f"Average Task Gain    : {np.mean(task_window):.4f}")
            print(f"Average Remote Ratio : {np.mean(remote_window):.4f}")
            print(
                f"Average Routing Compliance : "
                f"{100 * np.mean(routing_window):.2f}%"
            )
            if loss_history:
                print(f"Latest PPO Loss      : {loss_history[-1]:.4f}")
            print("=" * 60)

    if trainer.memory:
        loss = trainer.update()
        if loss is not None:
            loss_history.append(loss)

    print("\n========== FINAL RESULTS ==========")
    print(f"Total Episodes        : {len(reward_history)}")

    average_reward = float(np.mean(reward_history)) if reward_history else 0.0
    average_privacy = (
        float(np.mean(privacy_history)) if privacy_history else 0.0
    )
    average_task = (
        float(np.mean(task_gain_history)) if task_gain_history else 0.0
    )
    average_remote = (
        float(np.mean(remote_ratio_history))
        if remote_ratio_history
        else 0.0
    )
    average_routing_compliance = (
        float(np.mean(routing_compliance_history))
        if routing_compliance_history
        else 0.0
    )

    average_loss = (
        float(np.mean(loss_history)) if loss_history else 0.0
    )

    print(f"Average Reward        : {average_reward:.4f}")
    print(f"Average Privacy Leak  : {100 * average_privacy:.2f}%")
    print(f"Average Task Gain     : {average_task:.4f}")
    print(f"Average Remote Ratio  : {average_remote:.4f}")
    print(f"Average PPO Loss      : {average_loss:.4f}")
    print("===================================")

    results_path = Path(results_dir)
    raw_path = results_path / "raw"
    figures_path = results_path / "figures"
    models_path = results_path.parent / "models"

    for path in (raw_path, figures_path, models_path):
        path.mkdir(parents=True, exist_ok=True)

    training_csv_path = raw_path / "ppo_training.csv"
    with training_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "step",
                "reward",
                "privacy_leak",
                "task_gain",
                "remote_ratio",
                "routing_compliance",
            ]
        )
        writer.writerows(
            zip(
                step_history,
                reward_history,
                privacy_history,
                task_gain_history,
                remote_ratio_history,
                routing_compliance_history,
            )
        )

    summary = {
        "episodes": len(reward_history),
        "average_reward": average_reward,
        "average_privacy_leak": average_privacy,
        "average_task_gain": average_task,
        "average_remote_ratio": average_remote,
        "average_routing_compliance": average_routing_compliance,
        "average_ppo_loss": average_loss,
        "ppo_batch_size": ppo_batch_size,
        "ppo_epochs": ppo_epochs,
        "training_records": len(dataset),
        "seed": seed,
    }

    summary_path = raw_path / "ppo_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    model_path = models_path / "privacy_r1_agent.pt"
    torch.save(agent.state_dict(), model_path)

    print(f"Model saved to {model_path}")
    print(f"Training history saved to {training_csv_path}")
    print(f"Training summary saved to {summary_path}")

    return {
        "reward_history": reward_history,
        "privacy_history": privacy_history,
        "task_gain_history": task_gain_history,
        "remote_ratio_history": remote_ratio_history,
        "routing_compliance_history": routing_compliance_history,
        "loss_history": loss_history,
        "training_csv": str(training_csv_path),
        "summary_json": str(summary_path),
        "model_path": str(model_path),
    }
