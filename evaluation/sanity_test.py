from pathlib import Path
import sys
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.phase2_preprocessing import (
    initialize_chunker,
    MiniLMFeatureExtractor,
    chunk_text
)

from src.phase3_agent import PrivacyR1Agent

from src.phase4_reward import (
    calculate_privacy_leak,
    calculate_reward
)

def main():

    print("=" * 60)
    print("PRIVACY-R1 SANITY TEST")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")


    print("\n[1] Testing text chunking...")

    chunker = initialize_chunker()

    query = (
        "Patient David Chen has hypertension. "
        "His blood pressure is elevated. "
        "The doctor will adjust his medication."
    )

    chunks = chunk_text(
        query,
        chunker
    )

    assert len(chunks) > 0, (
        "Text chunking returned no chunks."
    )

    print(f"Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: {chunk}")


    print("\n[2] Testing MiniLM embeddings...")

    extractor = MiniLMFeatureExtractor(
        device=device
    )

    embeddings = extractor.get_embeddings(
        chunks
    )

    assert embeddings.shape[0] == len(chunks), (
        "Number of embeddings does not match "
        "number of chunks."
    )

    assert embeddings.shape[1] == 384, (
        "MiniLM embedding dimension should be 384."
    )

    print(
        f"Embedding shape: {tuple(embeddings.shape)}"
    )


    print("\n[3] Testing Privacy-R1 agent...")

    agent = PrivacyR1Agent(
        embedding_dim=384
    ).to(device)

    embeddings = embeddings.unsqueeze(0)

    with torch.no_grad():

        action_probs, state_values = agent(
            embeddings
        )

    expected_action_shape = (
        1,
        len(chunks),
        2
    )

    expected_value_shape = (
        1,
        len(chunks),
        1
    )

    assert (
        tuple(action_probs.shape)
        == expected_action_shape
    ), (
        f"Unexpected action shape: "
        f"{tuple(action_probs.shape)}"
    )

    assert (
        tuple(state_values.shape)
        == expected_value_shape
    ), (
        f"Unexpected value shape: "
        f"{tuple(state_values.shape)}"
    )

    print(
        f"Action probability shape: "
        f"{tuple(action_probs.shape)}"
    )

    print(
        f"State value shape: "
        f"{tuple(state_values.shape)}"
    )


    print("\n[4] Testing action sampling...")

    distribution = torch.distributions.Categorical(
        probs=action_probs
    )

    actions = distribution.sample()

    assert actions.shape == (
        1,
        len(chunks)
    ), (
        f"Unexpected action shape: "
        f"{tuple(actions.shape)}"
    )

    print(
        f"Actions: {actions.tolist()}"
    )

    for i, action in enumerate(actions[0].tolist()):

        route = (
            "LOCAL"
            if action == 0
            else "REMOTE"
        )

        print(
            f"  Chunk {i + 1}: {route}"
        )


    print("\n[5] Testing privacy metric...")

    pii_entities = [
        "David Chen"
    ]

    remote_chunks = [
        chunks[0]
    ]

    privacy_leak = calculate_privacy_leak(
        pii_entities,
        remote_chunks
    )

    assert 0.0 <= privacy_leak <= 1.0, (
        "Privacy leakage must be between 0 and 1."
    )

    print(
        f"Privacy leakage: "
        f"{privacy_leak:.4f}"
    )


    print("\n[6] Testing reward calculation...")

    task_gain = 1.0

    reward = calculate_reward(
        task_gain=task_gain,
        privacy_leak=privacy_leak
    )

    assert isinstance(
        reward,
        (float, int)
    ), (
        "Reward should be a numeric value."
    )

    print(
        f"Task gain: {task_gain:.4f}"
    )

    print(
        f"Privacy leakage: {privacy_leak:.4f}"
    )

    print(
        f"Final reward: {reward:.4f}"
    )


    print("\n" + "=" * 60)
    print("ALL SANITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
