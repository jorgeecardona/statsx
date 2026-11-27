"""Time and peak memory of one call, in statsx or fastumap, on made-up embeddings.

One call per process, so that the peak memory is that of the call:

    python scripts/compare_fastumap.py statsx kmeans 100000 --threads 4

It prints one JSON line. ``fastumap`` is needed only for the ``fastumap`` rows. The data are 8
Gaussian groups in float32; ``--float64`` converts them first. Run one measurement at a time on
a quiet machine, and compare the two libraries in the same run: the load of the machine moves the
seconds.
"""

# ruff: noqa: INP001, T201, ANN401 (a script that prints, and takes the results of two libraries)

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import resource
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

GROUPS = 8
CLUSTERS = 40
FASTUMAP_ITERATIONS = 50
BLOCK = 20_000


@dataclasses.dataclass(frozen=True)
class Run:
    """The settings of one measurement."""

    library: str
    task: str
    rows: int
    columns: int
    threads: int


def draw(run: Run, *, double: bool) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Return the data and the group of each row."""
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((GROUPS, run.columns)).astype(np.float32) * 0.5
    members = rng.integers(0, GROUPS, run.rows)
    x = centres[members]
    for start in range(0, run.rows, BLOCK):
        part = x[start : start + BLOCK]
        part += rng.standard_normal(part.shape, dtype=np.float32)
    return (x.astype(np.float64) if double else x), members


def calls(run: Run, x: npt.NDArray[Any]) -> dict[str, Callable[[], Any]]:
    """Return the call of each task in the chosen library."""
    if run.library == "fastumap":
        f: Any = importlib.import_module("fastumap")
        return {
            "umap": lambda: f.umap_project(x, 2),
            "kmeans": lambda: f.cluster(
                x, method="kmeans", n_clusters=CLUSTERS, n_iter=FASTUMAP_ITERATIONS
            ),
            "hdbscan": lambda: f.hdbscan(x, min_cluster_size=15),
            "elbow": lambda: f.choose_k(x, k_max=10),
        }
    from statsx.clustering import elbow_k, hdbscan, kmeans
    from statsx.embedding import umap

    threads = run.threads or None
    return {
        "umap": lambda: umap(x, threads=threads),
        "kmeans": lambda: kmeans(x, CLUSTERS, n_init=1, threads=threads),
        # the same number of iterations as the default of fastumap
        "kmeans50": lambda: kmeans(
            x, CLUSTERS, n_init=1, max_iter=FASTUMAP_ITERATIONS, tol=0.0, threads=threads
        ),
        "hdbscan": lambda: hdbscan(x, 15, threads=threads),
        "elbow": lambda: elbow_k(x, 10, threads=threads),
    }


def quality(
    run: Run, x: npt.NDArray[Any], members: npt.NDArray[np.int64], result: Any
) -> dict[str, Any]:
    """Return what shows that two timed calls did comparable work."""
    if run.task.startswith("kmeans"):
        labels = np.asarray(getattr(result, "labels", result))
        total, matched = 0.0, 0
        for c in np.unique(labels):
            part = x[labels == c].astype(np.float64)
            total += float(((part - part.mean(axis=0)) ** 2).sum())
            matched += int(np.bincount(members[labels == c]).max())
        found: dict[str, Any] = {"inertia": total / run.rows, "purity": matched / run.rows}
        if hasattr(result, "n_iter"):
            found |= {"n_iter": result.n_iter, "converged": result.converged}
        return found
    if run.task == "hdbscan":
        labels = np.asarray(getattr(result, "labels", result))
        return {"clusters": int(labels.max() + 1), "noise": float((labels < 0).mean())}
    if run.task == "elbow":
        return {"k": int(getattr(result, "k", result))}
    return {}


def main() -> None:
    """Measure one call and print it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library", choices=["statsx", "fastumap"])
    parser.add_argument("task", choices=["umap", "kmeans", "kmeans50", "hdbscan", "elbow"])
    parser.add_argument("rows", type=int)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4, help="0: the default of the library")
    parser.add_argument("--float64", action="store_true")
    given = parser.parse_args()
    run = Run(given.library, given.task, given.rows, given.columns, given.threads)
    x, members = draw(run, double=given.float64)
    table = calls(run, x)
    if run.task not in table:
        parser.error(f"{run.library} has no task {run.task}")
    held = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    result = table[run.task]()
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            dataclasses.asdict(run)
            | {
                "dtype": str(x.dtype),
                "seconds": round(seconds, 2),
                "peak_gb": round(peak / 1e6, 2),
                "data_gb": round(held / 1e6, 2),
                "load": round(os.getloadavg()[0], 1),
            }
            | quality(run, x, members, result)
        )
    )


if __name__ == "__main__":
    main()
