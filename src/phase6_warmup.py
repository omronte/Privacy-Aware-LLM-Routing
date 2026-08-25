
import torch
import torch.nn as nn
import torch.optim as optim


def generate_heuristic_labels(
    chunks: list[str],
    pii_entities: list[str]
) -> torch.Tensor:

    labels = []

    for chunk in chunks:
        chunk_lower = chunk.lower()

        has_pii = any(
            str(pii).lower() in chunk_lower
            for pii in pii_entities
        )

        labels.append(0.0 if has_pii else 1.0)

    return torch.tensor(
        labels,
        dtype=torch.float32
    )


def train_supervised_warmup(
    agent: nn.Module,
    dataloader: list,
    epochs: int = 1,
    lr: float = 3e-4
):

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if lr <= 0:
        raise ValueError("lr must be positive.")

    if not dataloader:
        raise ValueError(
            "Supervised warm-up dataloader is empty."
        )

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        agent.parameters(),
        lr=lr
    )

    agent.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for embeddings, labels in dataloader:

            optimizer.zero_grad()

            action_probs, _ = agent(embeddings)

            remote_probs = action_probs[:, :, 1]

            labels = labels.to(
                remote_probs.device
            )

            loss = criterion(
                remote_probs,
                labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Average BCE Loss: {average_loss:.4f}"
        )

    return agent
