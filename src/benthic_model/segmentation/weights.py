from __future__ import annotations

from collections import Counter


def inverse_freq_capped(
    labels: list[str],
    *,
    cap: float = 5.0,
    floor: float = 1.0,
) -> dict[str, float]:
    """Compute inverse-frequency class weights with clipping for stability."""
    if cap < floor:
        raise ValueError("cap must be >= floor")

    counts = Counter(labels)
    if not counts:
        raise ValueError("labels must not be empty")

    total = sum(counts.values())
    classes = len(counts)
    weights: dict[str, float] = {}
    for label, count in counts.items():
        raw = total / (classes * count)
        weights[label] = max(floor, min(cap, float(raw)))

    return weights
