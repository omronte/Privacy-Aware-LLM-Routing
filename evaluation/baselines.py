"""Deterministic routing baselines."""


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
        chunk_lower = chunk.lower()
        contains_pii = any(
            str(pii).strip().lower() in chunk_lower
            for pii in pii_entities
            if str(pii).strip()
        )
        actions.append(0 if contains_pii else 1)

    return actions
