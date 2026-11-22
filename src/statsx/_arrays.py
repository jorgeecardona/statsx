"""Argument checks shared by the modules that call the Rust kernels."""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

MAX_THREADS = 4
"""Threads used when ``threads`` is not given: a laptop shared with other work should not be
saturated by one call. The result does not depend on this number."""

MATRIX = 2
"""Dimensions of an array of points."""


def resolve_threads(threads: int | None) -> int:
    """Return the number of worker threads: ``threads``, or a default capped at ``MAX_THREADS``."""
    if threads is None:
        return max(1, min(MAX_THREADS, os.cpu_count() or 1))
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    return threads


def as_points(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``x`` as a C-contiguous float array of shape (n, d), n >= 2, with finite values."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != MATRIX or a.shape[0] < MATRIX or a.shape[1] < 1:
        raise ValueError("x must have shape (n, d) with at least 2 rows and 1 column")
    if not np.all(np.isfinite(a)):
        raise ValueError("x must be finite")
    return a
