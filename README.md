# Privacy-Aware LLM Routing

A privacy-aware LLM routing framework that dynamically routes sensitive and non-sensitive text chunks between local and remote LLM inference.

Inspired by the Privacy-R1 approach, this project combines PII detection, privacy-aware preprocessing, transformer-based actor-critic policy learning, supervised warm-up, and PPO reinforcement learning.

The routing objective is:

- **PII-containing chunks → LOCAL**
- **Non-PII chunks → REMOTE**

The system is designed to balance privacy protection, task performance, routing compliance, and practical remote LLM utilization.

## Why this project

Sending prompts to remote LLM providers can expose sensitive information when inputs contain personally identifiable information (PII).

This project addresses the problem through privacy-aware, chunk-level routing. Sensitive chunks are identified before routing, and a hard privacy constraint ensures that PII-sensitive chunks remain local while the learned policy determines routing behavior for suitable non-sensitive content.

The framework optimizes multiple objectives:

- Privacy leakage minimization
- Task performance
- Routing compliance
- Remote model utilization
---

## Overview

Sending user prompts to remote LLM providers can expose sensitive information when the input contains personally identifiable information (PII).

This project addresses the problem by processing an input prompt as multiple chunks and making privacy-aware routing decisions for each chunk.

The pipeline:

1. Preprocesses and chunks the input.
2. Detects PII and sensitive entities.
3. Generates embeddings using MiniLM.
4. Encodes chunk representations using a transformer-based policy.
5. Predicts routing actions using an actor-critic architecture.
6. Enforces a hard privacy constraint so PII-containing chunks remain local.
7. Computes a reward based on privacy leakage, task gain, and routing compliance.
8. Performs supervised warm-up.
9. Fine-tunes the routing policy using PPO.
10. Evaluates the trained policy using privacy and routing metrics.

---

## Key Features

- PII and sensitive entity detection
- Quasi-identifier detection
- Clinical-style named entity recognition
- Privacy-aware preprocessing and anonymization
- MiniLM-based chunk embeddings
- Transformer-based routing policy
- Actor-critic architecture
- Supervised policy warm-up
- PPO reinforcement learning
- Local/remote hybrid LLM routing
- Hard privacy constraint for PII chunks
- Privacy leakage evaluation
- Routing compliance evaluation
- Remote routing ratio analysis
- Task-gain evaluation
- Ablation analysis
- Training and evaluation visualizations
- Saved PPO routing-agent checkpoint

---

## Core Architecture

```text
                    Input Prompt
                         │
                         ▼
                Text Preprocessing
                         │
                         ▼
                     Chunking
                         │
                         ▼
              PII / Sensitive Entity
                    Detection
                         │
                         ▼
                 MiniLM Embeddings
                         │
                         ▼
              Transformer Encoder
                         │
                         ▼
             Actor-Critic Policy
                    │       │
                    │       └── Value / Critic
                    │
                    ▼
               Routing Action
                    │
          ┌─────────┴─────────┐
          │                   │
     PII / Sensitive       Non-PII
          │                   │
          ▼                   ▼
        LOCAL               REMOTE
        LLM                   LLM
          │                   │
          └─────────┬─────────┘
                    ▼
             Reward Function
                    │
                    ▼
               PPO Training
```

---

## Routing Policy

The routing policy is designed around a privacy-aware constraint:

```text
PII-containing chunk      → LOCAL
Non-PII chunk              → REMOTE
```

The learned policy optimizes the trade-off between:

- Privacy leakage
- Task gain
- Routing compliance
- Remote model utilization

A hard privacy constraint prevents PII-containing chunks from being intentionally routed to the remote model.

---

## Training Pipeline

### Phase 1 — Dataset Preparation

```text
src/phase1_dataset.py
```

Loads and prepares the training, validation, and test data.

---

### Phase 2 — Privacy Preprocessing

```text
src/phase2_preprocessing.py
```

Performs privacy-related preprocessing, entity detection, and sensitive-span handling.

---

### Phase 3 — Privacy-R1 Agent

```text
src/phase3_agent.py
```

Defines the transformer-based actor-critic routing policy.

The policy contains:

- Positional encoding
- Transformer encoder layers
- Actor head for routing actions
- Critic head for value estimation

---

### Phase 4 — Reward Function

```text
src/phase4_reward.py
```

Computes the reinforcement-learning reward using privacy and task-related signals.

The reward considers:

- Privacy leakage
- Task gain
- Routing compliance
- Remote usage

---

### Phase 5 — Hybrid Rollout

```text
src/phase5_rollout.py
```

Executes routing decisions and interacts with the local and remote LLM components.

---

### Phase 6 — Supervised Warm-Up

```text
src/phase6_warmup.py
```

Generates heuristic routing labels based on privacy information and provides an initial policy before PPO training.

---

### Phase 7 — PPO Training

```text
src/phase7_ppo.py
```

Fine-tunes the routing policy using **Proximal Policy Optimization (PPO)**.

---

# Repository Structure

```text
Privacy-Aware-LLM-Routing/
│
├── data/
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   └── gold_responses.json
│
├── evaluation/
│   ├── ablation.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── plot_results.py
│   ├── routing_test.py
│   ├── sanity_test.py
│   └── validate_dataset.py
│
├── models/
│   └── privacy_r1_agent.pt
│
├── results/
│   ├── figures/
│   │   ├── privacy_leak_curve.png
│   │   ├── remote_ratio_curve.png
│   │   ├── reward_curve.png
│   │   ├── routing_compliance_curve.png
│   │   └── task_gain_curve.png
│   │
│   ├── ablation.json
│   ├── ppo_summary.json
│   ├── ppo_training.csv
│   ├── routing_evaluation.json
│   ├── test_evaluation.json
│   ├── validation_evaluation.json
│   └── final_output.json
│
├── scripts/
│   └── prepare_clinical_data.py
│
├── src/
│   ├── phase1_dataset.py
│   ├── phase2_preprocessing.py
│   ├── phase3_agent.py
│   ├── phase4_reward.py
│   ├── phase5_rollout.py
│   ├── phase6_warmup.py
│   └── phase7_ppo.py
│
├── .env.example
├── .gitignore
├── main.py
├── Privacy_R1.ipynb
├── README.md
├── requirements.txt
└── test_remote.py
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Privacy-Aware-LLM-Routing.git
cd Privacy-Aware-LLM-Routing
```

---

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the required spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

# Hugging Face Token Setup

The project uses a Hugging Face token for remote model access.

## 1. Create `.env`

Copy the provided environment template.

### Windows PowerShell

```powershell
copy .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Then edit `.env`:

```env
HF_TOKEN=your_huggingface_token_here
```

Replace the placeholder with your own Hugging Face access token.

### Security

**Never commit your actual Hugging Face token.**

The repository intentionally contains:

```text
.env.example
```

but excludes:

```text
.env
```

through `.gitignore`.

The token is loaded through an environment variable rather than being hard-coded into the source code.

---

# Dataset

The repository contains:

```text
data/
├── train.json
├── validation.json
├── test.json
└── gold_responses.json
```

Each training/validation/test record is expected to contain the required fields:

```text
query
pii_units
target_response
```

The training pipeline validates the dataset schema before training.

The included data is intended for research and experimentation. Do not add real personally identifiable or confidential information to the repository.

---

# Running the Project

## Quick Smoke Test

Run a minimal training execution:

```bash
python main.py --steps 2 --warmup-epochs 1 --ppo-batch-size 2 --ppo-epochs 1
```

This is useful for verifying the pipeline and configuration.

---

## Standard Training Run

Example:

```bash
python main.py --steps 300 --warmup-epochs 1 --ppo-batch-size 8 --ppo-epochs 4
```

---

## Useful Arguments

```text
--train-file data/train.json
--skip-warmup
--no-debug-rollout
```

For all available arguments:

```bash
python main.py --help
```

---

# Evaluation

## 1. Full Pipeline Sanity Test

```bash
python evaluation/sanity_test.py
```

---

## 2. Routing Boundary Test

Tests the privacy routing boundary and verifies that PII-containing chunks are kept local.

```bash
python evaluation/routing_test.py
```

---

## 3. Policy Evaluation

```bash
python evaluation/evaluate.py data/validation.json --output results/validation_evaluation.json
```

---

## 4. Ablation Analysis

```bash
python evaluation/ablation.py data/test.json --output results/ablation.json
```

---

## 5. Dataset Validation

```bash
python evaluation/validate_dataset.py data/train.json
```

---

## 6. Generate Training Plots

```bash
python evaluation/plot_results.py
```

---

## 7. Test Remote Hugging Face Access

```bash
python test_remote.py
```

---

# Results

The repository includes the evaluation outputs and visualizations from the recorded experimental run.

## Privacy Leakage

![Privacy Leakage](results/figures/privacy_leak_curve.png)

Measures the privacy leakage observed during the recorded evaluation/training run.

---

## Remote Routing Ratio

![Remote Routing Ratio](results/figures/remote_ratio_curve.png)

Shows the proportion of routing decisions sent to the remote model during the recorded run.

---

## Reward

![Reward](results/figures/reward_curve.png)

Shows the reward progression during PPO training.

---

## Routing Compliance

![Routing Compliance](results/figures/routing_compliance_curve.png)

Measures compliance with the intended privacy-aware routing policy.

---

## Task Gain

![Task Gain](results/figures/task_gain_curve.png)

Shows the task-performance component of the routing objective during training.

> **Important:** These plots represent results from the included experimental run. They should not be interpreted as formal guarantees of privacy, utility, or generalization to unseen datasets.

---

# Trained Model

A trained routing-agent checkpoint is included at:

```text
models/privacy_r1_agent.pt
```

The checkpoint contains the learned parameters of the transformer-based actor-critic routing agent, including:

- Positional encoding
- Transformer encoder layers
- Actor head
- Critic head

The checkpoint can be used for evaluation without retraining the routing policy from scratch.

---

# Output Files

Training and evaluation outputs include:

```text
models/privacy_r1_agent.pt

results/
├── figures/
│   ├── privacy_leak_curve.png
│   ├── remote_ratio_curve.png
│   ├── reward_curve.png
│   ├── routing_compliance_curve.png
│   └── task_gain_curve.png
│
├── ablation.json
├── ppo_summary.json
├── ppo_training.csv
├── routing_evaluation.json
├── test_evaluation.json
├── validation_evaluation.json
└── final_output.json
```

---

# Technologies

- Python
- PyTorch
- Transformers
- Hugging Face Hub
- Sentence Transformers
- spaCy
- Microsoft Presidio
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Privacy and Security

This project focuses on privacy-aware LLM routing.

Important security practices include:

- PII detection before routing
- Hard local routing for PII-containing chunks
- Environment-variable based credential management
- `.env` excluded from version control
- `.env.example` provided for configuration
- No API tokens embedded in source code
- Synthetic/research data used for repository examples

### Important limitation

PII detection systems can produce false positives and false negatives. Therefore, the routing mechanism should not be interpreted as a universal privacy guarantee.

For production deployment, additional privacy controls, auditing, access controls, encryption, and security testing would be required.

---

# Reproducibility

Results can vary between runs because of:

- PPO stochasticity
- Random initialization
- GPU execution
- Remote model/provider behavior
- Provider availability and quotas
- Dataset differences
- Model version changes

For meaningful comparisons, use the same:

- Dataset splits
- Model configuration
- Training parameters
- Evaluation configuration

The included checkpoint and evaluation outputs are provided to make the reported experiment easier to reproduce.

---

# Troubleshooting

### Hugging Face token error

If you receive an error indicating that the Hugging Face token is missing:

1. Make sure `.env` exists.
2. Check that it contains:

```env
HF_TOKEN=your_actual_token
```

3. Make sure the token is valid.
4. Run:

```bash
python test_remote.py
```

---

### Missing spaCy model

Run:

```bash
python -m spacy download en_core_web_sm
```

---

### Remote provider/API error

Remote inference may fail because of:

- Invalid credentials
- Provider availability
- Account limitations
- Model availability
- Network issues
- Provider quotas

Check your Hugging Face account and model/provider access.

---

# Project Status

This repository contains a research-oriented implementation of privacy-aware LLM routing, including:

- Privacy preprocessing
- Transformer-based routing
- Supervised warm-up
- PPO fine-tuning
- Hybrid local/remote inference
- Evaluation
- Ablation analysis
- Training visualizations
- Trained routing-agent checkpoint

The implementation is intended for research, experimentation, and educational use.

---

# License

No open-source license has currently been specified.

If this project is released publicly for reuse, an appropriate license should be added to the repository.
