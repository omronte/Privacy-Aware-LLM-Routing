"""Deterministic routing baselines."""

from src.privacy import contains_pii


def all_local_actions(chunks):
    """Route every chunk to the local model."""
    return [0] * len(chunks)


def all_remote_actions(chunks):
    """Route every chunk to the remote model.

    This intentionally ignores privacy and is useful as a leakage upper-bound
    baseline. Do not use it for production inference.
    """
    return [1] * len(chunks)


def heuristic_actions(chunks, pii_entities):
    """Keep chunks containing detected PII local; route other chunks remotely."""
    actions = []

    for chunk in chunks:
        actions.append(0 if contains_pii(chunk, pii_entities) else 1)

    return actions
