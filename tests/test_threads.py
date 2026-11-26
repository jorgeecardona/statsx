"""The default number of threads, concurrent calls, and the same result in another process."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from statsx import _arrays
from statsx._arrays import available_cpus, cpu_quota, resolve_threads
from statsx.clustering import dbscan, gaussian_mixture, hdbscan, kmeans
from statsx.embedding import umap


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


@pytest.mark.parametrize(
    ("membership", "files", "expected"),
    [
        # cgroup v2 inside a container: the group is the mount point
        ("0::/\n", {"cpu.max": "50000 100000\n"}, 0.5),
        # v2 on a host: the smallest limit on the way up wins
        (
            "0::/a/b\n",
            {"a/cpu.max": "150000 100000\n", "a/b/cpu.max": "400000 100000\n"},
            1.5,
        ),
        ("0::/a\n", {"a/cpu.max": "max 100000\n"}, None),
        # cgroup v1
        (
            "4:cpu,cpuacct:/docker/x\n",
            {
                "cpu,cpuacct/cpu.cfs_quota_us": "200000\n",
                "cpu,cpuacct/cpu.cfs_period_us": "100000\n",
            },
            2.0,
        ),
        (
            "3:cpu:/g\n",
            {"cpu/g/cpu.cfs_quota_us": "-1\n", "cpu/g/cpu.cfs_period_us": "100000\n"},
            None,
        ),
        ("5:memory:/g\n", {"cpu.max": "50000 100000\n"}, None),
        ("0::/\n", {"cpu.max": "not a number\n"}, None),
        ("0::/\n", {}, None),
    ],
)
def test_the_cpu_limit_is_read_from_the_control_group(
    tmp_path: Path, membership: str, files: dict[str, str], expected: float | None
) -> None:
    _tree(tmp_path / "fs", files)
    (tmp_path / "cgroup").write_text(membership)
    assert cpu_quota(tmp_path / "fs", tmp_path / "cgroup") == expected


def test_a_system_without_control_groups_has_no_limit(tmp_path: Path) -> None:
    assert cpu_quota(tmp_path, tmp_path / "missing") is None


@pytest.mark.parametrize(
    ("cpus", "quota", "expected"),
    [(16, None, 16), (16, 0.5, 1), (16, 1.5, 1), (16, 3.0, 3), (2, 8.0, 2), (1, None, 1)],
)
def test_available_cpus_is_the_smaller_of_affinity_and_the_limit(
    monkeypatch: pytest.MonkeyPatch, cpus: int, quota: float | None, expected: int
) -> None:
    def affinity(pid: int) -> set[int]:
        del pid
        return set(range(cpus))

    monkeypatch.setattr(os, "sched_getaffinity", affinity, raising=False)
    monkeypatch.setattr(_arrays, "cpu_quota", lambda: quota)
    assert available_cpus() == expected
    monkeypatch.delenv("STATSX_THREADS", raising=False)
    assert resolve_threads(None) == min(4, expected)


def test_the_variable_sets_the_default_and_an_argument_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATSX_THREADS", "9")
    assert resolve_threads(None) == 9
    assert resolve_threads(2) == 2
    for bad in ("0", "-1", "two", "1.5"):
        monkeypatch.setenv("STATSX_THREADS", bad)
        with pytest.raises(ValueError, match="STATSX_THREADS"):
            resolve_threads(None)
    monkeypatch.setenv("STATSX_THREADS", " ")
    assert 1 <= resolve_threads(None) <= 4
    with pytest.raises(ValueError, match="threads must be at least 1"):
        resolve_threads(0)


_PROGRAM = """
import hashlib
import numpy as np
from statsx.clustering import dbscan, gaussian_mixture, hdbscan, kmeans
from statsx.embedding import umap

x = np.random.default_rng(0).normal(size=(700, 12))
x[:200] += 4.0
fit = umap(x, n_epochs=40)
results = [
    kmeans(x, 3).centroids,
    kmeans(x, 3, metric="cosine").centroids,
    gaussian_mixture(x, 2).means,
    hdbscan(x, 15).mst,
    dbscan(x, 4.0).labels,
    fit.embedding,
    fit.transform(x[:30]).embedding,
]
print(hashlib.sha256(b"".join(np.ascontiguousarray(r).tobytes() for r in results)).hexdigest())
"""


def test_two_processes_give_the_same_bytes() -> None:
    digests = {
        subprocess.run(  # noqa: S603 (this interpreter and a program written above)
            [sys.executable, "-c", _PROGRAM],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "STATSX_THREADS": threads},
        ).stdout.strip()
        for threads in ("1", "3")
    }
    assert len(digests) == 1
    assert len(digests.pop()) == hashlib.sha256().digest_size * 2


def test_calls_from_several_threads_give_what_one_thread_gives() -> None:
    rng = np.random.default_rng(1)
    data = [rng.normal(size=(500, 8)) + 3.0 * (rng.uniform(size=(500, 1)) < 0.4) for _ in range(4)]

    def run(index: int) -> tuple[bytes, ...]:
        x = data[index % 4]
        return (
            kmeans(x, 3).centroids.tobytes(),
            gaussian_mixture(x, 2).means.tobytes(),
            hdbscan(x, 10).mst.tobytes(),
            dbscan(x, 2.5).labels.tobytes(),
            umap(x, n_epochs=20).embedding.tobytes(),
        )

    alone = [run(i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        together = list(pool.map(run, range(16)))
    assert all(together[i] == alone[i % 4] for i in range(16))
