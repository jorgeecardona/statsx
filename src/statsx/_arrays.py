"""Argument checks and the cosine distance, shared by the modules that call the Rust kernels."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import numpy.typing as npt

MAX_THREADS = 4
"""Threads used when ``threads`` is not given: a laptop shared with other work should not be
saturated by one call. The result does not depend on this number."""

MATRIX = 2
"""Dimensions of an array of points."""

Metric = Literal["euclidean", "cosine"]
"""The distances that ``umap`` and the clustering functions accept for the points."""

METRICS = ("euclidean", "cosine")

SAME_DIRECTION = 1e-14
"""Unit rows closer than this are one direction: it is rounding of the normalised coordinates."""


def check_metric(metric: str) -> None:
    """Raise ``ValueError`` unless ``metric`` is a known distance."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def unit_rows(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``a`` with each row scaled to length 1, and refuse a row of zeros."""
    scale = np.abs(a).max(axis=1)
    if np.any(scale == 0.0):
        raise ValueError("the cosine distance is undefined for a row of zeros")
    scaled = a / scale[:, None]
    return np.ascontiguousarray(scaled / np.sqrt(np.einsum("ij,ij->i", scaled, scaled))[:, None])


def cosine_of_chord(distance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``1 - cos`` for unit vectors at Euclidean ``distance``, which is ``distance^2 / 2``.

    A distance within rounding of 0 gives 0 exactly.
    """
    return np.where(distance <= SAME_DIRECTION, 0.0, distance * distance / 2.0)


def resolve_threads(threads: int | None) -> int:
    """Return the number of worker threads: ``threads``, or a default capped at ``MAX_THREADS``."""
    if threads is None:
        return max(1, min(MAX_THREADS, os.cpu_count() or 1))
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    return threads


def as_points(x: npt.ArrayLike, *, min_rows: int = MATRIX) -> npt.NDArray[np.float64]:
    """Return ``x`` as a C-contiguous float array of shape (n, d), n >= ``min_rows``, finite."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != MATRIX or a.shape[0] < min_rows or a.shape[1] < 1:
        raise ValueError(f"x must have shape (n, d) with at least {min_rows} rows and 1 column")
    if not np.all(np.isfinite(a)):
        raise ValueError("x must be finite")
    return a
