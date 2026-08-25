"""Fast unit test for the Phase 5 privacy routing boundary."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.phase5_rollout import generate_hybrid_response


def main():
    chunks = [
        "My name is John Smith and my phone number is 9876543210.",
        "Explain supervised learning.",
    ]
    pii_entities = ["John Smith", "9876543210"]

    calls = {"local": [], "remote": []}

    def fake_local(text):
        calls["local"].append(text)
        return f"LOCAL[{text}]"

    def fake_remote(text):
        calls["remote"].append(text)
        return f"REMOTE[{text}]"

    response, remote_chunks, actual_actions, forced = generate_hybrid_response(
        chunks=chunks,
        actions=[1, 1],
        pii_entities=pii_entities,
        local_model_fn=fake_local,
        remote_model_fn=fake_remote,
    )

    assert actual_actions == [0, 1], (
        f"Expected [LOCAL, REMOTE], got {actual_actions}"
    )
    assert forced == [True, False], f"Unexpected forced flags: {forced}"
    assert remote_chunks == [chunks[1]], (
        "PII chunk was incorrectly sent to remote model."
    )
    assert calls["remote"] == [chunks[1]], (
        f"Unexpected remote calls: {calls['remote']}"
    )
    assert calls["local"] == [chunks[0]], (
        f"Unexpected local calls: {calls['local']}"
    )
    assert response.startswith("LOCAL[")
    assert response.endswith("REMOTE[Explain supervised learning.]")

    print("=" * 60)
    print("PHASE 5 PRIVACY ROUTING TEST")
    print("=" * 60)
    print("PPO requested     : [REMOTE, REMOTE]")
    print("Actual routing    : [LOCAL, REMOTE]")
    print("PII forced local  : [True, False]")
    print("Remote chunks     :", remote_chunks)
    print("Result            : PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
