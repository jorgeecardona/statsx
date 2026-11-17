"""Rank helper shared by the rank tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def midranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the ranks of ``values`` from 1, with the mean rank for tied values."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks
