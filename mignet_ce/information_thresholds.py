from __future__ import annotations

"""Shared thresholds for judging whether macro future information is informative."""

import math


def closure_signal_threshold_bits(k: int) -> float:
    """Return the canonical low-signal threshold for a K-state macro target."""
    if int(k) < 1:
        raise ValueError("k must be positive when resolving closure signal threshold.")
    return max(0.01, 0.01 * math.log2(max(int(k), 2)))
