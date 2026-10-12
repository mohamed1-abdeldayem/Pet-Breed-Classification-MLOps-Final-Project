from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping


def calculate_psi(
    baseline: Mapping[str, int],
    current: Iterable[str],
    epsilon: float = 1e-6,
) -> float:
    """Compare categorical distributions using the Population Stability Index."""
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")

    current_counts = Counter(current)
    baseline_total = sum(baseline.values())
    current_total = sum(current_counts.values())

    if baseline_total <= 0:
        raise ValueError("baseline must contain at least one observation")
    if current_total <= 0:
        raise ValueError("current observations must not be empty")
    if any(value < 0 for value in baseline.values()):
        raise ValueError("baseline counts cannot be negative")

    categories = set(baseline) | set(current_counts)
    psi = 0.0

    for category in categories:
        expected = max(baseline.get(category, 0) / baseline_total, epsilon)
        actual = max(current_counts.get(category, 0) / current_total, epsilon)
        psi += (actual - expected) * math.log(actual / expected)

    return float(psi)

