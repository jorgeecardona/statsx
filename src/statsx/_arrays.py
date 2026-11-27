"""Argument checks and the cosine distance, shared by the modules that call the Rust kernels."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

MAX_THREADS = 4
"""Most threads used when ``threads`` is not given: a laptop shared with other work should not be
saturated by one call. The result does not depend on this number."""

THREADS_VARIABLE = "STATSX_THREADS"
"""Environment variable that sets the default number of threads."""

CGROUP_ROOT = Path("/sys/fs/cgroup")
CGROUP_MEMBERSHIP = Path("/proc/self/cgroup")

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


def _ratio(quota: Path, period: Path | None = None) -> float | None:
    """Return quota / period from cgroup files, or None when there is no limit or no file.

    With one file (cgroup v2, ``cpu.max``) it holds both numbers, or ``max`` for no limit. With
    two (v1, ``cpu.cfs_quota_us`` and ``cpu.cfs_period_us``) a quota of -1 is no limit.
    """
    try:
        words = quota.read_text().split()
        if period is not None:
            words = [words[0], period.read_text().split()[0]]
        used, every = float(words[0]), float(words[1])
    except (OSError, ValueError, IndexError):
        return None
    return used / every if used > 0 and every > 0 else None


def cpu_quota(root: Path = CGROUP_ROOT, membership: Path = CGROUP_MEMBERSHIP) -> float | None:
    """Return the CPU time limit of this process's control group, in CPUs, or None.

    Reads the limit of the group and of every group above it, and returns the smallest. Works
    with cgroup v1 and v2. None means no limit can be seen, which is also the case on systems
    without control groups, and on hosts that limit a container outside it (AWS Fargate).
    """
    try:
        lines = membership.read_text().splitlines()
    except OSError:
        return None
    limits: list[float] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:  # noqa: PLR2004
            continue
        controllers, path = parts[1], parts[2].strip("/")
        if controllers == "":  # v2: one tree
            bases, v2 = [root], True
        elif "cpu" in controllers.split(","):  # v1: the cpu controller has its own tree
            bases, v2 = [root / controllers, root / "cpu"], False
        else:
            continue
        for base in bases:
            node = base / path
            # the group, then each group above it up to the mount point; inside a container the
            # path may not exist under the mount, and then only the mount point answers
            for folder in (node, *node.parents):
                found = (
                    _ratio(folder / "cpu.max")
                    if v2
                    else _ratio(folder / "cpu.cfs_quota_us", folder / "cpu.cfs_period_us")
                )
                if found is not None:
                    limits.append(found)
                if folder == base:
                    break
    return min(limits) if limits else None


def available_cpus() -> int:
    """Return how many CPUs this process may use, at least 1.

    The smaller of the CPUs it may be scheduled on and the whole CPUs of its control group's
    time limit: a limit of 1.5 CPUs gives 1, so that threads are not throttled.
    """
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:  # macOS and Windows
        count = os.cpu_count() or 1
    quota = cpu_quota()
    if quota is not None:
        count = min(count, math.floor(quota))
    return max(1, count)


def resolve_threads(threads: int | None) -> int:
    """Return the number of worker threads to use.

    ``threads`` if given. Otherwise the value of ``STATSX_THREADS`` if it is set. Otherwise the
    CPUs the process may use, at most ``MAX_THREADS``.

    Raises:
        ValueError: If ``threads``, or the variable, is not a whole number of at least 1.
    """
    if threads is None:
        setting = os.environ.get(THREADS_VARIABLE)
        if setting is None or not setting.strip():
            return min(MAX_THREADS, available_cpus())
        if not setting.strip().isdigit() or int(setting) < 1:
            raise ValueError(f"{THREADS_VARIABLE} must be a whole number >= 1, got {setting!r}")
        return int(setting)
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    return threads


Rows = npt.NDArray[np.float32] | npt.NDArray[np.float64]
"""Points in the type the kernels read where they are."""

_CHECK_BLOCK = 4096
"""Rows checked for finite values at a time, so the check needs little memory."""


def _check_points(a: npt.NDArray[np.floating], min_rows: int) -> None:
    """Raise ``ValueError`` unless ``a`` is finite, (n, d), with n >= ``min_rows`` and d >= 1."""
    if a.ndim != MATRIX or a.shape[0] < min_rows or a.shape[1] < 1:
        raise ValueError(f"x must have shape (n, d) with at least {min_rows} rows and 1 column")
    for start in range(0, a.shape[0], _CHECK_BLOCK):
        if not np.all(np.isfinite(a[start : start + _CHECK_BLOCK])):
            raise ValueError("x must be finite")


def as_points(x: npt.ArrayLike, *, min_rows: int = MATRIX) -> npt.NDArray[np.float64]:
    """Return ``x`` as a C-contiguous float array of shape (n, d), n >= ``min_rows``, finite."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    _check_points(a, min_rows)
    return a


def contiguous(x: npt.ArrayLike) -> Rows:
    """Return ``x`` as a C-contiguous array: float32 when it is float32, else float64.

    A C-contiguous array of either type is returned as it is, not copied.
    """
    given = np.asarray(x)
    if given.dtype == np.float32:
        return np.ascontiguousarray(given, dtype=np.float32)
    return np.ascontiguousarray(given, dtype=np.float64)


def as_rows(x: npt.ArrayLike, *, min_rows: int = MATRIX) -> Rows:
    """``as_points`` that keeps float32 data as float32."""
    a = contiguous(x)
    _check_points(a, min_rows)
    return a
