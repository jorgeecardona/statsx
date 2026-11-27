"""Fewer coordinates per point: UMAP maps and principal component analysis (PCA).

``umap`` follows McInnes, Healy & Melville (2018). It has two phases. First it builds a weighted
graph: each point is joined to its ``n_neighbors`` nearest points, the weight of an edge falls with
the distance beyond the nearest neighbour (section 3.1 of the paper), and the two directions of an
edge are merged by the probabilistic sum ``a + b - a b``. Then it places the points in
``dimensions`` dimensions so that points joined by heavy edges sit close and unjoined points sit
apart. It starts from the eigenvectors of the normalized graph Laplacian (Algorithm 4) and moves
the points by stochastic gradient descent on the fuzzy set cross entropy with negative sampling
(section 4.1, Algorithm 5).

Everything is written in Rust and NumPy: the neighbour search is exact (statsx has no
approximate index; ``umap(x, neighbors=table)`` takes the table of one and ``neighbor_recall``
measures it on a sample of rows), the eigenvectors come from LOBPCG, the curve ``a``, ``b`` from
Levenberg-Marquardt, and the layout from the algorithm of the paper. A result does not depend on
the number of threads.

A picture of a map is not evidence of groups. UMAP keeps the neighbours of each point and gives up
distances and densities (section 6 of the paper): the size of a cluster on the map, the gap
between two clusters and the order of the clusters mean nothing. UMAP also "can tend to find
manifold structure within the noise of a dataset" (section 6), most of all with few points.
``neighbor_preservation`` measures how many of the nearest neighbours the map keeps, and
``UMAPFit.cross_entropy`` is the value of the objective that the layout lowers. Cluster the data,
not the map, when the groups matter.

The distance can be the Euclidean one or the cosine one, ``1 - cos``. The paper builds the graph
from the distances to the neighbours of each point and nothing else (section 3.1), so any
dissimilarity can feed it. The cosine distance is not a metric (it breaks the triangle
inequality), and the paper states its theory for metric spaces, so the cosine map is a use of the
construction beyond what the paper proves.

``UMAPFit.transform`` places new points in a finished map. The paper lists this as future work
(section 7) and no paper describes it (Reichmann et al. 2024 say so, section 3.2.1). It is
Algorithm 5 run for the new points alone with the training points held fixed, the method that the
uwot vignette on embedding new data describes, and umap-learn follows. It is a statsx
implementation of that description and not a published algorithm: the docs say so.

A map can be refreshed: ``umap(x, init=positions)`` runs Algorithm 5 from positions that the
caller gives, in place of the spectral start of Algorithm 4. The first step size of Algorithm 5,
which the paper sets to 1, is ``learning_rate``; LargeVis writes the same linear schedule with the
first value as a parameter. The start decides the large-scale arrangement of a map (Kobak &
Linderman 2021), so two maps that share a start can be compared point by point. A point whose data
did not change still moves a little.

Known labels can guide a map: ``umap(x, labels=y, label_weight=w)`` lowers the weight between
neighbours with different labels. The paper names this as future work (section 7) and no paper
specifies it, so it is a statsx construction: the label view with weight 1 for equal labels and
``1 - w`` for different ones, intersected with the graph of the data by the product, before the
union. de Ridder et al. (2003, eq. 3) add to the distance between classes in another method, which
is the nearest published idea. A map built with labels is not evidence that the data separates
them: see ``label_agreement``.

``pca`` projects points onto the directions along which they vary most (Pearson 1901; Hotelling
1933). The projection is the linear one with the smallest squared reconstruction error. It is
exact and uses NumPy's eigendecomposition. Use it to look at the linear structure, or to reduce
wide data before clustering or ``umap``. It keeps the directions of largest variance, which are
not always the directions that separate groups.

References:
    Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.
    *Philosophical Magazine* 2(11):559-572. https://doi.org/10.1080/14786440109462720.

    Hotelling, H. (1933). Analysis of a complex of statistical variables into principal
    components. *Journal of Educational Psychology* 24(6):417-441.
    https://doi.org/10.1037/h0071325.

    McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and
    Projection for Dimension Reduction. arXiv:1802.03426 (version 3, 2020).
    https://arxiv.org/abs/1802.03426.

    Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver: locally optimal block
    preconditioned conjugate gradient method. *SIAM Journal on Scientific Computing* 23(2):517-541.
    https://doi.org/10.1137/S1064827500366124.

    Duersch, J. A., Shao, M., Yang, C., & Gu, M. (2018). A robust and efficient implementation of
    LOBPCG. *SIAM Journal on Scientific Computing* 40(5):C655-C676.
    https://doi.org/10.1137/17M1129830.

    de Ridder, D., Kouropteva, O., Okun, O., Pietikäinen, M., & Duin, R. P. W. (2003). Supervised
    locally linear embedding. *Artificial Neural Networks and Neural Information Processing,
    ICANN/ICONIP 2003*, Lecture Notes in Computer Science 2714:333-341.
    https://doi.org/10.1007/3-540-44989-2_40.

    Kobak, D., & Linderman, G. C. (2021). Initialization is critical for preserving global data
    structure in both t-SNE and UMAP. *Nature Biotechnology* 39:156-157.
    https://doi.org/10.1038/s41587-020-00809-z.

    Levenberg, K. (1944). A method for the solution of certain non-linear problems in least
    squares. *Quarterly of Applied Mathematics* 2(2):164-168.

    Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters.
    *Journal of the Society for Industrial and Applied Mathematics* 11(2):431-441.

    Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). Visualizing large-scale and high-dimensional
    data. *Proceedings of the 25th International Conference on World Wide Web*, pp. 287-297.
    https://doi.org/10.1145/2872427.2883041.

    Reichmann, L., Hägele, D., & Weiskopf, D. (2024). Out-of-core dimensionality reduction for
    large data via out-of-sample extensions. arXiv:2408.04129. States that no paper describes the
    out-of-sample extension of UMAP. https://arxiv.org/abs/2408.04129.

    uwot documentation, "Embedding New Data" (article of the R package uwot, version 0.2.5.9000).
    Describes the placement of new points in a fixed map.
    https://jlmelville.github.io/uwot/articles/embedding-new-data.html.
"""

from __future__ import annotations

import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

import numpy as np
import numpy.typing as npt

from statsx import _accel
from statsx._arrays import (
    Metric,
    as_points,
    check_metric,
    cosine_of_chord,
    resolve_threads,
    unit_rows,
)

_PCA_BLOCK = 4096
"""Rows per block of ``pca``, so that no second copy of the data is held."""

_FORMAT = 4
"""Version of the ``.npz`` layout of ``UMAPFit.to_npz``. Raise it when the layout changes."""

_PAPER_RATE = 1.0
"""First step size of Algorithm 5 in the paper: the default without ``init``."""

_REFRESH_RATE = 0.1
"""First step size by default with ``init``. A choice of statsx, from no paper (D-78)."""

Start = Literal["spectral", "init"]
"""Where the layout started: the eigenvectors of Algorithm 4, or positions given by the caller."""

NeighborSource = Literal["exact", "given"]
"""Where the neighbours came from: the exact search of statsx, or a table of the caller."""

_TABLE_TOLERANCE = 1e-3
"""Largest accepted gap between a given distance and the recomputed one, relative to that
distance plus the median distance of the table. It allows the rounding of float32."""

_RECALL_SAMPLE = 1000
"""Rows that ``neighbor_recall`` checks by default."""

_WORST_SHARE = 10
"""``NeighborRecall.worst_decile`` is the mean over this share (1 in 10) of the sampled rows."""

_MIN_TRANSFORM_EPOCHS = 30
"""Fewest epochs of ``transform`` by default."""

_TRANSFORM_SHARE = 3
"""``transform`` runs one third of the epochs of the fit by default."""

_CURVE_END = 3.0
"""The curve of the paper is fitted on distances in (0, 3]: ``Psi`` has fallen to ``exp(-3)``."""

_CURVE_POINTS = 300
"""Distances on that grid, spaced evenly."""

_SPECTRAL_ITERATIONS = 1000
"""Largest number of LOBPCG iterations for the start."""

_SPECTRAL_TOLERANCE = 1e-6
"""Residual norm below which an eigenvector counts as found."""

_BOX = 10.0
"""The start is scaled so that its largest coordinate is this far from the origin."""

_CELL = 25.0
"""Distance between the centres of two connected components in the start."""

_JITTER = 1e-4
"""Standard deviation of the noise added to the start, so that no two points coincide."""

_LARGE = 10_000
"""Points above which the default number of epochs drops from 500 to 200."""

_LM_ITERATIONS = 200

_FARTHEST = 1e100
"""Largest size of a coordinate of ``init``: squared distances must not overflow."""

_MIN_NEIGHBORS = 2
"""Fewest neighbours: the weights of one neighbour always sum to 1, so sigma is undefined."""

_UNLABELLED = -1
"""The label of a point that has none."""


@dataclass(frozen=True, slots=True)
class FuzzyGraph:
    """The weighted graph of the points: the 1-skeleton of the fuzzy simplicial set.

    The graph is symmetric and stored in compressed rows. The weights are in (0, 1], read as the
    probability that the two points are joined.

    Args:
        indptr: Row starts, shape (n + 1,). Row ``i`` holds ``indices[indptr[i]:indptr[i + 1]]``.
        indices: The column of each entry, ascending within a row.
        weights: The weight of each entry.
    """

    indptr: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]

    @property
    def n_vertices(self) -> int:
        """The number of points."""
        return int(self.indptr.shape[0]) - 1

    @property
    def n_edges(self) -> int:
        """The number of undirected edges."""
        return int(self.indices.shape[0]) // 2

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        """The row of each entry, shape (2 * n_edges,)."""
        return np.repeat(np.arange(self.n_vertices, dtype=np.int64), np.diff(self.indptr))

    def dense(self) -> npt.NDArray[np.float64]:
        """The (n, n) weight matrix. It holds n^2 values: use it for small n only."""
        n = self.n_vertices
        out = np.zeros((n, n))
        out[self.rows, self.indices] = self.weights
        return out


@dataclass(frozen=True, slots=True)
class Neighbors:
    """The nearest other rows of each row of a data set.

    Args:
        indices: Row numbers, shape (n, k), nearest first, ties by row number.
        distances: The distance to each of those in ``metric``, shape (n, k).
        metric: ``"euclidean"``, or ``"cosine"`` for ``1 - cos``.
    """

    indices: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float64]
    metric: Metric


NeighborTable = npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike]
"""Neighbours that the caller brings: row numbers (n, k), or row numbers and distances."""


@dataclass(frozen=True, slots=True)
class Supervision:
    """The labels that changed the graph of a map.

    Args:
        labels: The label of each point, shape (n,): 0 or above, and -1 for a point without one.
        label_weight: The share taken from the weight of a neighbour with another label, in
            [0, 1). 0 leaves the graph as it is without labels.
        cross_label_share: Of the neighbour weight between two labelled points, before the labels
            changed it, the share that joined different labels: how mixed the labels are among
            neighbours in the data. None when no two labelled points are neighbours.
    """

    labels: npt.NDArray[np.int64] = field(repr=False)
    label_weight: float
    cross_label_share: float | None


@dataclass(frozen=True, slots=True)
class NeighborRecall:
    """How many true nearest neighbours a table holds, measured on a sample of rows.

    Args:
        recall: Mean over the sampled rows of the share of their ``k`` neighbours in the table
            that are as near as the true ``k``-th nearest. 1 means no neighbour is missing.
        standard_error: Standard error of ``recall`` from the sampling of rows, without
            replacement. 0 when every row was checked.
        worst_decile: Mean over the tenth of the sampled rows with the lowest share.
        per_row: The share of each sampled row, shape (m,).
        rows: The sampled row numbers, ascending, shape (m,).
        k: Neighbours per row that were checked.
    """

    recall: float
    standard_error: float
    worst_decile: float
    per_row: npt.NDArray[np.float64] = field(repr=False)
    rows: npt.NDArray[np.int64] = field(repr=False)
    k: int

    @property
    def n_sample(self) -> int:
        """The number of rows checked."""
        return int(self.rows.shape[0])


@dataclass(frozen=True, slots=True)
class UMAPPlacement:
    """New points placed in a finished map by ``UMAPFit.transform``.

    Args:
        embedding: The position of each new point, shape (m, dimensions).
        neighbors: The ``n_neighbors`` nearest training points of each new point, shape (m, k),
            as row numbers of the fitted data, nearest first.
        distances: The distance to each of those in the metric of the fit, shape (m, k). The first
            column is how far the new point is from the training data: compare it with the same
            column of ``UMAPFit.distances``. A new point much farther out than the training points
            lands where the map has no information.
        sigma: The scale of each new point: the ``sigma`` with ``sum exp(-d / sigma) = log2(k)``.
        n_epochs: Passes of the placement.
        seed: The seed of the random sampling.
    """

    embedding: npt.NDArray[np.float64]
    neighbors: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    n_epochs: int
    seed: int

    @property
    def n_points(self) -> int:
        """The number of new points."""
        return int(self.embedding.shape[0])


@dataclass(frozen=True, slots=True)
class UMAPFit:
    """A UMAP map and what it was built from.

    Args:
        embedding: The position of each point, shape (n, dimensions).
        graph: The weighted graph of the points.
        data: The points that the map was built from, shape (n, d). ``transform`` finds the
            neighbours of new points among them.
        neighbors: The ``n_neighbors`` nearest other points of each point, shape (n, k), nearest
            first.
        distances: The distance to each of those in ``metric``, shape (n, k).
        metric: ``"euclidean"``, or ``"cosine"`` for ``1 - cos`` of the angle between two rows.
        rho: The distance from each point to its nearest neighbour with a distance above zero.
        sigma: The scale of each point, solved so that the weights of its neighbours sum to
            ``log2(n_neighbors)``.
        a: The ``a`` of ``1 / (1 + a r^(2 b))``, the membership of a pair at distance ``r`` in the map.
        b: The ``b`` of the same curve.
        min_dist: The distance below which the curve is fitted to 1.
        n_epochs: Passes of the layout.
        negative_sample_rate: Negative samples per sampled edge.
        seed: The seed of the start and the layout. With ``init`` it seeds the layout only.
        n_graph_components: Connected components of the graph. A spectral start treats each one
            on its own.
        n_random_start: Points whose component was too small for a spectral start and got a
            random one.
        spectral_converged: True when every spectral start reached the tolerance. When False the
            start is the last iterate, which is still a smooth map of the graph but not the
            eigenvector. Also True when no spectral start ran (``start`` is ``"init"``).
        learning_rate: The first step size of the layout.
        start: ``"spectral"``, or ``"init"`` when the caller gave the start positions.
        neighbor_source: ``"exact"``, or ``"given"`` when the caller gave the neighbours. A
            given table has the quality that ``neighbor_recall`` measures and no other claim.
        supervision: The labels given to ``umap`` and what they changed, or None for a map of
            the data alone. With it, ``graph`` is not the graph of the data: ``neighbors``,
            ``distances``, ``rho`` and ``sigma`` still are.
    """

    embedding: npt.NDArray[np.float64]
    graph: FuzzyGraph
    data: npt.NDArray[np.float64]
    neighbors: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float64]
    metric: Metric
    rho: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    a: float
    b: float
    min_dist: float
    n_epochs: int
    negative_sample_rate: int
    seed: int
    n_graph_components: int
    n_random_start: int
    spectral_converged: bool
    learning_rate: float
    start: Start
    neighbor_source: NeighborSource
    supervision: Supervision | None

    @property
    def n_points(self) -> int:
        """The number of points."""
        return int(self.embedding.shape[0])

    @property
    def dimensions(self) -> int:
        """The dimension of the map."""
        return int(self.embedding.shape[1])

    @property
    def n_neighbors(self) -> int:
        """The number of neighbours of each point."""
        return int(self.neighbors.shape[1])

    def transform(
        self,
        x: npt.ArrayLike,
        *,
        n_epochs: int | None = None,
        seed: int | None = None,
        threads: int | None = None,
    ) -> UMAPPlacement:
        """Place new points in this map without changing it.

        Each new point is joined to its ``n_neighbors`` nearest training points with the weights
        of section 3.1 of the paper, except that ``rho`` is 0: the weight is ``exp(-d / sigma)``,
        so the nearest point does not get weight 1 by force. uwot and umap-learn do the same for
        new points (their local connectivity is lowered by one). The point starts at the weighted
        mean of its neighbours' positions. Then it runs Algorithm 5: its edges attract it, and
        training points drawn at random repel it. The training points never move and the new
        points do not act on each other. The random stream of a point is seeded by its row number
        in ``x``.

        No paper describes this step (the paper lists it as future work, section 7). This is the
        method of the uwot vignette on embedding new data, which umap-learn follows, written
        again from that description. The neighbours are exact, and the result does not depend on
        ``threads``.

        Args:
            x: The new points, shape (m, d) with the same ``d`` as the fitted data, finite.
            n_epochs: Passes of the placement. 0 returns the weighted mean of the neighbours. The
                default is a third of the epochs of the fit, at least 30. This is a choice of
                statsx.
            seed: Seed of the random sampling. Defaults to the seed of the fit.
            threads: Worker threads. Defaults to the CPUs the process may use, at most 4, or to
                the ``STATSX_THREADS`` environment variable.

        Raises:
            ValueError: If ``x`` has a different number of columns than the fitted data, if
                ``n_epochs`` is negative, or if the fit used the cosine distance and a row of
                ``x`` is all zeros.
        """
        new = as_points(x, min_rows=1)
        if new.shape[1] != self.data.shape[1]:
            raise ValueError(
                f"x has {new.shape[1]} columns, the fitted data has {self.data.shape[1]}"
            )
        epochs = (
            max(_MIN_TRANSFORM_EPOCHS, self.n_epochs // _TRANSFORM_SHARE)
            if n_epochs is None
            else n_epochs
        )
        if epochs < 0:
            raise ValueError(f"n_epochs must be at least 0, got {epochs}")
        stream = self.seed if seed is None else seed
        workers = resolve_threads(threads)
        idx, dist = _neighbors_between(self.data, new, self.n_neighbors, self.metric, workers)
        position, sigma = _accel.umap_place(
            self.embedding,
            idx,
            dist,
            self.a,
            self.b,
            epochs,
            self.negative_sample_rate,
            stream,
            workers,
        )
        return UMAPPlacement(
            embedding=position,
            neighbors=idx,
            distances=dist,
            sigma=sigma,
            n_epochs=epochs,
            seed=stream,
        )

    def to_npz(self, file: str | os.PathLike[str] | IO[bytes]) -> None:
        """Save the fit to a ``.npz`` file that ``UMAPFit.from_npz`` reads back.

        The file holds NumPy arrays and no pickled objects, so loading it runs no code. Every
        value is stored exactly: a loaded fit places new points exactly as this one does. The
        file keeps the fitted data, so it is as large as the data. The layout has a version
        number, and ``from_npz`` refuses a file of another version.

        Args:
            file: A path, written as given (no ``.npz`` is added), or a binary file object.
        """
        arrays = {
            "format_version": np.int64(_FORMAT),
            "embedding": self.embedding,
            "data": self.data,
            "neighbors": self.neighbors,
            "distances": self.distances,
            "rho": self.rho,
            "sigma": self.sigma,
            "graph_indptr": self.graph.indptr,
            "graph_indices": self.graph.indices,
            "graph_weights": self.graph.weights,
            "reals": np.array(
                [self.a, self.b, self.min_dist, self.learning_rate], dtype=np.float64
            ),
            "counts": np.array(
                [
                    self.n_epochs,
                    self.negative_sample_rate,
                    self.n_graph_components,
                    self.n_random_start,
                    int(self.spectral_converged),
                ],
                dtype=np.int64,
            ),
            "seed": np.uint64(self.seed),
            "metric": np.array(self.metric),
            "start": np.array(self.start),
            "neighbor_source": np.array(self.neighbor_source),
        }
        if self.supervision is not None:
            share = self.supervision.cross_label_share
            arrays["labels"] = self.supervision.labels
            arrays["label_weight"] = np.asarray(self.supervision.label_weight, np.float64)
            arrays["cross_label_share"] = np.asarray(
                math.nan if share is None else share, np.float64
            )
        if isinstance(file, (str, os.PathLike)):
            with Path(file).open("wb") as handle:
                np.savez(handle, **arrays)  # type: ignore[arg-type]
        else:
            np.savez(file, **arrays)  # type: ignore[arg-type]

    @classmethod
    def from_npz(cls, file: str | os.PathLike[str] | IO[bytes]) -> UMAPFit:
        """Load a fit saved by ``UMAPFit.to_npz``.

        A file of format version 1 holds no ``learning_rate`` and no ``start``: every such fit
        used 1.0 and a spectral start, and loads with those values. A file of version 1 or 2
        holds no ``neighbor_source``: every such fit searched its own neighbours. No file of
        version 1 to 3 had labels.

        The arrays are checked before they are used: their shapes agree, the neighbours and the
        graph point at existing rows, and the values are finite. A file of another format
        version, or with a missing or inconsistent array, raises ``ValueError``.

        Args:
            file: A path or a binary file object.

        Raises:
            ValueError: If the file is not a fit that this version wrote, or is inconsistent.
        """
        try:
            with np.load(file, allow_pickle=False) as store:
                found = {name: store[name] for name in store.files}
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError(f"not a UMAPFit file: {error}") from error
        version = int(found["format_version"]) if "format_version" in found else None
        if version is None or version not in range(1, _FORMAT + 1):
            raise ValueError(f"the file has format version {version}, this reads 1 to {_FORMAT}")
        added = ("start", "neighbor_source")[: version - 1]
        supervised = version >= _FIRST_LABELLED and any(name in found for name in _LABEL_FIELDS)
        fields = (*_FIELDS, *added, *(_LABEL_FIELDS if supervised else ()))
        missing = sorted(set(fields) - set(found))
        if missing:
            raise ValueError(f"the file lacks {missing}")
        reals = _floats(found["reals"], 1)
        counts = _ints(found["counts"], 1)
        if reals.shape != (3 if version == 1 else 4,) or counts.shape != (5,):
            raise ValueError("the arrays in the file have shapes that do not fit together")
        start = str(found["start"]) if "start" in added else "spectral"
        source = str(found["neighbor_source"]) if "neighbor_source" in added else "exact"
        metric = str(found["metric"])
        check_metric(metric)
        fit = cls(
            embedding=_floats(found["embedding"], 2),
            graph=FuzzyGraph(
                _ints(found["graph_indptr"], 1),
                _ints(found["graph_indices"], 1),
                _floats(found["graph_weights"], 1),
            ),
            data=_floats(found["data"], 2),
            neighbors=_ints(found["neighbors"], 2),
            distances=_floats(found["distances"], 2),
            metric=metric,  # type: ignore[arg-type]
            rho=_floats(found["rho"], 1),
            sigma=_floats(found["sigma"], 1),
            a=float(reals[0]),
            b=float(reals[1]),
            min_dist=float(reals[2]),
            n_epochs=int(counts[0]),
            negative_sample_rate=int(counts[1]),
            seed=int(found["seed"]),
            n_graph_components=int(counts[2]),
            n_random_start=int(counts[3]),
            spectral_converged=bool(counts[4]),
            learning_rate=_PAPER_RATE if version == 1 else float(reals[3]),
            start=start,  # type: ignore[arg-type]
            neighbor_source=source,  # type: ignore[arg-type]
            supervision=_loaded_supervision(found) if supervised else None,
        )
        _check_fit(fit)
        fit.data.flags.writeable = False
        return fit

    def neighbor_recall(
        self, *, sample: int = _RECALL_SAMPLE, seed: int = 0, threads: int | None = None
    ) -> NeighborRecall:
        """Measure the neighbours of this fit against exact ones, on a sample of rows.

        It is the module function ``neighbor_recall`` on the data, the neighbours and the metric
        of the fit. It is 1 for a fit whose neighbours statsx searched.

        Args:
            sample: Rows to check, at least 2. All rows when there are fewer.
            seed: Seed of the sample.
            threads: Worker threads. The result does not depend on it.
        """
        return neighbor_recall(
            self.data,
            self.neighbors,
            metric=self.metric,
            sample=sample,
            seed=seed,
            threads=threads,
        )

    def cross_entropy(self, *, threads: int | None = None) -> float:
        """Return the fuzzy set cross entropy of the map (Definition 10 of the paper).

        It is summed over all pairs of points, with weight 0 for a pair that is not an edge. It is
        0 when the map reproduces the graph exactly, and the layout lowers it. It costs
        O(n^2 d), so it is not stored.

        Args:
            threads: Worker threads. The value does not depend on it.
        """
        return float(
            _accel.umap_cross_entropy(
                self.embedding,
                self.graph.indptr,
                self.graph.indices,
                self.graph.weights,
                self.a,
                self.b,
                resolve_threads(threads),
            )
        )


_FIELDS = (
    "embedding",
    "data",
    "neighbors",
    "distances",
    "rho",
    "sigma",
    "graph_indptr",
    "graph_indices",
    "graph_weights",
    "reals",
    "counts",
    "seed",
    "metric",
)
"""The arrays that a saved fit holds besides its format version."""

_LABEL_FIELDS = ("labels", "label_weight", "cross_label_share")
"""The arrays that a fit with labels holds besides those."""

_FIRST_LABELLED = 4
"""The first format version that can hold labels."""


def _numeric(value: npt.NDArray[np.generic], ndim: int) -> None:
    """Raise ``ValueError`` unless ``value`` is a numeric array with ``ndim`` dimensions."""
    if value.ndim != ndim or value.dtype.kind not in "fiu":
        raise ValueError(f"expected a {ndim}-dimensional numeric array, got shape {value.shape}")


def _floats(value: npt.NDArray[np.generic], ndim: int) -> npt.NDArray[np.float64]:
    """Return ``value`` as a C-contiguous float array of ``ndim`` dimensions."""
    _numeric(value, ndim)
    return np.ascontiguousarray(value, dtype=np.float64)


def _ints(value: npt.NDArray[np.generic], ndim: int) -> npt.NDArray[np.int64]:
    """Return ``value`` as a C-contiguous integer array of ``ndim`` dimensions."""
    _numeric(value, ndim)
    return np.ascontiguousarray(value, dtype=np.int64)


def _loaded_supervision(found: dict[str, npt.NDArray[np.generic]]) -> Supervision:
    """Return the labels of a saved fit; ``_check_fit`` checks their values."""
    labels = found["labels"]
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("the labels in the file are not a list of integers")
    weight, share = found["label_weight"], found["cross_label_share"]
    _numeric(weight, 0)
    _numeric(share, 0)
    return Supervision(
        labels=np.ascontiguousarray(labels, dtype=np.int64),
        label_weight=float(weight),
        cross_label_share=None if math.isnan(float(share)) else float(share),
    )


def _sound_supervision(supervision: Supervision | None, n: int) -> bool:
    """Whether the labels of a loaded fit are ones that ``umap`` can have recorded."""
    if supervision is None:
        return True
    share = supervision.cross_label_share
    return (
        supervision.labels.shape == (n,)
        and bool(np.all(supervision.labels >= _UNLABELLED))
        and 0.0 <= supervision.label_weight < 1.0
        and (share is None or 0.0 <= share <= 1.0)
    )


def _check_fit(fit: UMAPFit) -> None:
    """Raise ``ValueError`` unless the parts of a loaded fit agree with each other."""
    n, k = fit.neighbors.shape
    dim = fit.embedding.shape[1]
    g = fit.graph
    shapes_agree = (
        fit.embedding.shape[0] == n
        and fit.data.shape[0] == n
        and fit.distances.shape == (n, k)
        and fit.rho.shape == (n,)
        and fit.sigma.shape == (n,)
        and g.indptr.shape == (n + 1,)
        and g.indices.shape == g.weights.shape
        and k >= _MIN_NEIGHBORS
        and dim >= 1
    )
    if not shapes_agree:
        raise ValueError("the arrays in the file have shapes that do not fit together")
    finite = all(
        bool(np.all(np.isfinite(v)))
        for v in (fit.embedding, fit.data, fit.distances, fit.rho, fit.sigma, g.weights)
    )
    sound_graph = (
        g.indptr[0] == 0
        and g.indptr[-1] == g.indices.shape[0]
        and bool(np.all(np.diff(g.indptr) >= 0))
        and bool(np.all((g.indices >= 0) & (g.indices < n)))
        and bool(np.all((g.weights > 0.0) & (g.weights <= 1.0)))
    )
    ordered = np.sort(fit.neighbors, axis=1)
    sound_neighbors = bool(
        np.all((fit.neighbors >= 0) & (fit.neighbors < n))
        & np.all(ordered[:, 1:] != ordered[:, :-1])
        & np.all(fit.neighbors != np.arange(n, dtype=np.int64)[:, None])
    )
    sound_settings = (
        fit.a > 0.0
        and fit.b > 0.0
        and 0.0 <= fit.min_dist <= 1.0
        and fit.n_epochs >= (0 if fit.start == "init" else 1)
        and fit.negative_sample_rate >= 0
        and 0.0 < fit.learning_rate <= _PAPER_RATE
        and fit.start in {"spectral", "init"}
        and fit.neighbor_source in {"exact", "given"}
        and (fit.start == "spectral" or fit.n_random_start == 0)
    )
    sound_labels = _sound_supervision(fit.supervision, n)
    if not (finite and sound_graph and sound_neighbors and sound_settings and sound_labels):
        raise ValueError("the file holds values that a UMAPFit cannot have")


def fit_curve(min_dist: float) -> tuple[float, float]:
    """Return ``(a, b)`` so that ``1 / (1 + a r^(2 b))`` follows the curve of the paper.

    The curve is 1 for ``r <= min_dist`` and ``exp(-(r - min_dist))`` beyond it (Definition 11).
    The fit is nonlinear least squares on 300 evenly spaced distances in (0, 3], by the
    Levenberg-Marquardt method with Marquardt's scaling, on ``log a`` and ``log b`` so that both
    stay positive. The paper does not say how the fit is done or on which distances.

    Args:
        min_dist: The distance in [0, 1] below which two points in the map count as joined.
    """
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError(f"min_dist must be in [0, 1], got {min_dist}")
    r = np.linspace(0.0, _CURVE_END, _CURVE_POINTS)[1:]
    target = np.where(r <= min_dist, 1.0, np.exp(-(r - min_dist)))
    log_r = np.log(r)

    def model(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, b = np.exp(p)
        return 1.0 / (1.0 + a * np.exp(2.0 * b * log_r))

    p = np.zeros(2)
    damping = 1e-3
    residual = model(p) - target
    sse = float(residual @ residual)
    gain = 0.0
    for _ in range(_LM_ITERATIONS):
        a, b = np.exp(p)
        t = a * np.exp(2.0 * b * log_r)
        slope = -1.0 / (1.0 + t) ** 2
        jac = np.column_stack([slope * t, slope * t * 2.0 * b * log_r])
        gradient = jac.T @ residual
        hessian = jac.T @ jac
        improved = False
        while damping < 1e12:  # noqa: PLR2004 (give up when the step is hopeless)
            step = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian)), -gradient)
            trial = model(p + step) - target
            trial_sse = float(trial @ trial)
            if trial_sse < sse:
                gain = sse - trial_sse
                p, residual, sse = p + step, trial, trial_sse
                damping = max(damping / 3.0, 1e-12)
                improved = True
                break
            damping *= 4.0
        if not improved or gain <= 1e-15 * sse:
            break
    a, b = np.exp(p)
    return float(a), float(b)


def _sub_graph(graph: FuzzyGraph, members: npt.NDArray[np.int64]) -> FuzzyGraph:
    """The graph on ``members``, a union of connected components, renumbered from 0."""
    position = np.full(graph.n_vertices, -1, dtype=np.int64)
    position[members] = np.arange(members.shape[0], dtype=np.int64)
    counts = graph.indptr[members + 1] - graph.indptr[members]
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    first = np.repeat(graph.indptr[members] - indptr[:-1], counts)
    entries = first + np.arange(int(indptr[-1]), dtype=np.int64)
    return FuzzyGraph(indptr, position[graph.indices[entries]], graph.weights[entries])


def _start(
    graph: FuzzyGraph, dimensions: int, seed: int, threads: int
) -> tuple[npt.NDArray[np.float64], int, int, bool]:
    """Return the start, the number of components, the points started at random and convergence.

    Each connected component gets the eigenvectors of its own normalized Laplacian after the
    trivial one (Algorithm 4), scaled to a box that holds a share of the volume equal to the
    share of the points. A component with fewer than ``dimensions + 2`` points has no such
    eigenvectors and starts at random. The components are placed on a grid. The paper only warns
    that disconnected graphs need care (section 4.3).
    """
    rng = np.random.default_rng(seed)
    n = graph.n_vertices
    labels = _accel.umap_components(graph.indptr, graph.indices)
    n_components = int(labels.max()) + 1
    members = [np.flatnonzero(labels == c) for c in range(n_components)]
    order = sorted(range(n_components), key=lambda c: (-members[c].shape[0], c))
    side = int(np.ceil(n_components ** (1.0 / dimensions)))
    largest = members[order[0]].shape[0]
    start = np.empty((n, dimensions))
    random_points = 0
    converged = True
    for rank, c in enumerate(order):
        idx = members[c]
        sub = graph if n_components == 1 else _sub_graph(graph, idx)
        found = None
        if idx.shape[0] >= dimensions + 2:
            found = _accel.umap_spectral(
                sub.indptr,
                sub.indices,
                sub.weights,
                dimensions,
                _SPECTRAL_ITERATIONS,
                _SPECTRAL_TOLERANCE,
                seed,
                threads,
            )
        if found is None:
            coords = rng.uniform(-1.0, 1.0, size=(idx.shape[0], dimensions))
            random_points += idx.shape[0]
        else:
            coords, _, ok, _ = found
            converged = converged and bool(ok)
        peak = float(np.max(np.abs(coords)))
        scale = _BOX * (idx.shape[0] / largest) ** (1.0 / dimensions)
        centre = np.array(np.unravel_index(rank, (side,) * dimensions), dtype=np.float64) * _CELL
        start[idx] = coords * (scale / peak if peak > 0.0 else 0.0) + centre
    start += rng.normal(0.0, _JITTER, size=start.shape)
    return start, n_components, random_points, converged


def _given_start(init: npt.ArrayLike, n: int, dimensions: int) -> npt.NDArray[np.float64]:
    """Return a copy of ``init``, checked as the start of a layout of ``n`` points."""
    start = np.array(init, dtype=np.float64, order="C", copy=True)
    if start.shape != (n, dimensions):
        raise ValueError(
            f"init must have shape (n, dimensions) = {(n, dimensions)}, got {start.shape}"
        )
    if not np.all(np.isfinite(start)) or np.max(np.abs(start)) > _FARTHEST:
        raise ValueError(f"init must be finite and at most {_FARTHEST:g} in size")
    if np.linalg.matrix_rank(start - start.mean(axis=0)) < dimensions:
        raise ValueError(
            f"the rows of init lie in fewer than {dimensions} dimensions (all on one point, line "
            "or plane): every force is along a difference of positions, so the layout would stay "
            "there"
        )
    return start


def _neighbors(
    a: npt.NDArray[np.float64], k: int, metric: Metric, workers: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the ``k`` nearest other rows of each row and their distances in ``metric``.

    Two unit rows at Euclidean distance ``e`` have cosine distance ``e^2 / 2``, so the nearest
    points are the same for both. The distances are not: ``e^2 / 2`` is what the graph needs.
    Rows that are the same direction up to rounding get distance 0 exactly.
    """
    if metric == "euclidean":
        return _accel.umap_knn(a, k, workers)
    idx, euclid = _accel.umap_knn(unit_rows(a), k, workers)
    return idx, cosine_of_chord(euclid)


def _neighbors_between(
    reference: npt.NDArray[np.float64],
    query: npt.NDArray[np.float64],
    k: int,
    metric: Metric,
    workers: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the ``k`` nearest rows of ``reference`` for each row of ``query``, and distances.

    The cosine distance is handled as in ``_neighbors``.
    """
    if metric == "euclidean":
        return _accel.umap_knn_between(reference, query, k, workers)
    idx, euclid = _accel.umap_knn_between(unit_rows(reference), unit_rows(query), k, workers)
    return idx, cosine_of_chord(euclid)


def _given_neighbors(
    a: npt.NDArray[np.float64], table: NeighborTable, k: int | None, metric: Metric, workers: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the ``k`` nearest of the neighbours in ``table`` of each row, and their distances.

    A row's own number is dropped wherever it stands. The distances are computed from ``a``, and
    each row is ordered by distance, then row number. ``k`` None takes as many as every row has.
    Distances in ``table`` are only compared with the computed ones.
    """
    n = a.shape[0]
    stated: npt.NDArray[np.float64] | None = None
    raw = table
    if isinstance(table, tuple):
        if len(table) != 2:  # noqa: PLR2004 (indices and distances)
            raise ValueError("neighbors as a tuple must be (indices, distances)")
        raw, stated = table[0], np.asarray(table[1], dtype=np.float64)
    idx = np.asarray(raw)
    if idx.dtype.kind not in "iu":
        raise ValueError(f"neighbors must be whole row numbers, got dtype {idx.dtype}")
    if idx.ndim != 2 or idx.shape[0] != n or idx.shape[1] < 1:  # noqa: PLR2004 (a table)
        raise ValueError(f"neighbors must have shape ({n}, k), got {idx.shape}")
    outside = (idx < 0) | (idx >= n)
    if np.any(outside):
        raise ValueError(
            f"neighbors holds {idx[outside][0]}, which is not a row of x (some libraries write "
            "-1 when they find too few neighbours)"
        )
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    ordered = np.sort(idx, axis=1)
    twice = np.flatnonzero(np.any(ordered[:, 1:] == ordered[:, :-1], axis=1))
    if twice.size:
        raise ValueError(f"row {twice[0]} of neighbors names a row twice")
    every = np.arange(n, dtype=np.int64)
    own = idx == every[:, None]
    others = int(idx.shape[1] - own.sum(axis=1).max())
    if others < 1 or (k is not None and others < k):
        raise ValueError(
            f"neighbors needs at least {k or 1} other rows in every row, and one row has {others}"
        )
    points = a if metric == "euclidean" else unit_rows(a)
    squared = _accel.umap_table_d2(points, every, idx, workers)
    if stated is not None:
        _check_stated(stated, np.sqrt(squared), own, metric)
    squared = np.where(own, np.inf, squared)  # a row is not its own neighbour: sort it last
    first = np.lexsort((idx, squared), axis=-1)[:, : others if k is None else k]
    near = np.take_along_axis(idx, first, axis=1)
    chosen = np.take_along_axis(squared, first, axis=1)
    return np.ascontiguousarray(near), _in_metric(chosen, metric)


def _in_metric(squared: npt.NDArray[np.float64], metric: Metric) -> npt.NDArray[np.float64]:
    """Return squared distances of rows (of unit rows for cosine) as distances in ``metric``."""
    chord = np.sqrt(squared)
    return np.ascontiguousarray(chord if metric == "euclidean" else cosine_of_chord(chord))


def _check_stated(
    stated: npt.NDArray[np.float64],
    chord: npt.NDArray[np.float64],
    own: npt.NDArray[np.bool_],
    metric: Metric,
) -> None:
    """Raise ``ValueError`` unless the distances of a given table are the ones of the data."""
    true = chord if metric == "euclidean" else cosine_of_chord(chord)
    if stated.shape != true.shape:
        raise ValueError(
            f"the distances of neighbors have shape {stated.shape}, the row numbers {true.shape}"
        )
    other = ~own
    given, true = stated[other], true[other]
    allowed = _TABLE_TOLERANCE * (true + float(np.median(true))) + 1e-6

    def agrees(expected: npt.NDArray[np.float64]) -> bool:
        return bool(np.all(np.abs(given - expected) <= allowed))

    if agrees(true):
        return
    hint = ""
    squares = np.abs(given - true**2) <= _TABLE_TOLERANCE * (true**2 + 1.0)
    if metric == "euclidean" and np.all(squares):
        hint = " They are squared distances: pass their square root."
    if metric == "cosine" and np.all(np.abs(given - (1.0 - true)) <= _TABLE_TOLERANCE):
        hint = " They are cosine similarities: pass 1 minus each."
    raise ValueError(
        f"the distances of neighbors are not the {metric} distances between those rows of x.{hint}"
    )


def _labels(labels: npt.ArrayLike, n: int, lowest: int, name: str) -> npt.NDArray[np.int64]:
    """Return ``n`` integer labels of at least ``lowest`` as a new read-only array."""
    given = np.asarray(labels)
    if given.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {given.shape}")
    if given.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers, got {given.dtype}: encode the classes as 0, 1")
    out = np.array(given, dtype=np.int64)
    if np.any(out < lowest):
        raise ValueError(f"{name} must be {lowest} or above")
    out.flags.writeable = False
    return out


def _supervision_settings(
    labels: npt.ArrayLike | None, label_weight: float | None, n: int
) -> tuple[npt.NDArray[np.int64], float] | None:
    """Return the checked labels and weight, or None when neither is given."""
    if labels is None:
        if label_weight is not None:
            raise ValueError("label_weight needs labels")
        return None
    if label_weight is None:
        raise ValueError("labels needs label_weight, from 0 (no effect) to below 1")
    if not 0.0 <= label_weight < 1.0:
        raise ValueError(f"label_weight must be at least 0 and below 1, got {label_weight}")
    return _labels(labels, n, _UNLABELLED, "labels"), float(label_weight)


def nearest_neighbors(
    x: npt.ArrayLike, k: int, *, metric: Metric = "euclidean", threads: int | None = None
) -> Neighbors:
    """Return the ``k`` nearest other rows of each row of ``x``, by exact search.

    ``umap`` runs this search itself. Call it to build the table once and pass it to several
    ``umap`` calls as ``neighbors``. The cost is O(n^2 d).

    Args:
        x: The points, shape (n, d), finite.
        k: Neighbours per row, from 1 to n - 1.
        metric: ``"euclidean"``, or ``"cosine"`` for ``1 - cos``.
        threads: Worker threads. The result does not depend on it.

    Raises:
        ValueError: If ``k`` is not in [1, n - 1], or ``metric`` is ``"cosine"`` and a row is
            all zeros.
    """
    check_metric(metric)
    a = as_points(x)
    n = a.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n - 1] = [1, {n - 1}], got {k}")
    idx, dist = _neighbors(a, k, metric, resolve_threads(threads))
    return Neighbors(indices=idx, distances=dist, metric=metric)


def _sample_rows(n: int, sample: int, seed: int) -> npt.NDArray[np.int64]:
    """Return ``sample`` different row numbers below ``n``, ascending; all rows when fewer."""
    if sample < 2:  # noqa: PLR2004 (a standard error needs two rows)
        raise ValueError(f"sample must be at least 2, got {sample}")
    if sample >= n:
        return np.arange(n, dtype=np.int64)
    drawn = np.random.default_rng(seed).choice(n, size=sample, replace=False)
    return np.sort(drawn).astype(np.int64)


def _neighbors_of(
    a: npt.NDArray[np.float64], rows: npt.NDArray[np.int64], k: int, metric: Metric, workers: int
) -> npt.NDArray[np.int64]:
    """Return the ``k`` nearest other rows of ``a`` for each of ``rows``, by exact search."""
    idx, _ = _neighbors_between(a, np.ascontiguousarray(a[rows]), k + 1, metric, workers)
    # drop the row itself; when copies of it crowd it out of the list, drop the last entry
    others = np.argsort(idx == rows[:, None], axis=1, kind="stable")[:, :k]
    return np.take_along_axis(idx, others, axis=1)


def neighbor_recall(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    neighbors: NeighborTable,
    *,
    k: int | None = None,
    metric: Metric = "euclidean",
    sample: int = _RECALL_SAMPLE,
    seed: int = 0,
    threads: int | None = None,
) -> NeighborRecall:
    """Measure how many true nearest neighbours a table holds, on a sample of rows.

    For each sampled row it finds the exact ``k`` nearest other rows among all rows, and counts
    the neighbours in the table (its ``k`` nearest, as ``umap`` would take them) that are at most
    as far as the ``k``-th of those. A swap between rows at the same distance, such as copies of
    one row, counts as found. The cost is O(sample n d). The result is the mean over a random
    sample of rows, a choice of statsx: it comes from no paper.

    Args:
        x: The points, shape (n, d), finite.
        neighbors: Row numbers, shape (n, K), or ``(indices, distances)``. See ``umap``.
        k: Neighbours per row to check. The default is as many as every row of the table has.
        metric: The distance between rows of ``x``.
        sample: Rows to check, at least 2. All rows when there are fewer.
        seed: Seed of the sample.
        threads: Worker threads. The result does not depend on it.

    Raises:
        ValueError: If the table is not one that ``umap`` accepts, or ``k`` or ``sample`` is out
            of range.
    """
    check_metric(metric)
    a = as_points(x)
    n = a.shape[0]
    if k is not None and not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n - 1] = [1, {n - 1}], got {k}")
    workers = resolve_threads(threads)
    rows = _sample_rows(n, sample, seed)
    near, dist = _given_neighbors(a, neighbors, k, metric, workers)
    size = int(near.shape[1])
    truth = _neighbors_of(a, rows, size, metric, workers)
    # the same function measures both tables, so equal distances compare equal
    points = a if metric == "euclidean" else unit_rows(a)
    exact = _in_metric(_accel.umap_table_d2(points, rows, truth, workers), metric)
    per_row = (dist[rows] <= exact.max(axis=1, keepdims=True)).mean(axis=1)
    m = int(rows.shape[0])
    spread = float(per_row.std(ddof=1))
    return NeighborRecall(
        recall=float(per_row.mean()),
        standard_error=spread * float(np.sqrt((1.0 - m / n) / m)),
        worst_decile=float(np.sort(per_row)[: -(-m // _WORST_SHARE)].mean()),
        per_row=per_row,
        rows=rows,
        k=size,
    )


def umap(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    *,
    n_neighbors: int = 15,
    dimensions: int = 2,
    metric: Metric = "euclidean",
    min_dist: float = 0.1,
    n_epochs: int | None = None,
    negative_sample_rate: int = 5,
    init: npt.ArrayLike | None = None,
    learning_rate: float | None = None,
    neighbors: NeighborTable | None = None,
    labels: npt.ArrayLike | None = None,
    label_weight: float | None = None,
    seed: int = 0,
    threads: int | None = None,
) -> UMAPFit:
    """Return a UMAP map of the rows of ``x``.

    Without ``neighbors`` the search is exact, so the cost is O(n^2 d) for the neighbours and
    O(n_neighbors n n_epochs) for the layout. The result does not depend on ``threads``. A different ``seed`` gives a different
    map: the layout is a stochastic search for a minimum, not a unique answer.

    The default ``n_epochs`` is 500 up to 10,000 points and 200 above. The paper leaves the number
    to the caller. ``negative_sample_rate`` 5 is the value of LargeVis (Tang et al. 2016), whose
    sampling the paper follows.

    ``init`` refreshes a map: the layout starts at the positions given, used exactly as they are
    (no rescaling, no noise) and copied, so the caller's array is never written. The graph is
    built from ``x`` as usual. Rows need positions that fit their data: for a window that moves
    over time, ``init=old.transform(x).embedding`` places every row of ``x`` in the old map. A row
    whose data changed and whose start was left at its old position may not reach its new place at
    a ``learning_rate`` below 1. A row whose data did not change still moves a little: to know how
    much, run ``umap(old.data, init=old.embedding)`` and measure the movement.

    ``learning_rate`` is the first step size of the layout, which falls linearly to 0. It is 1.0
    without ``init``, the value of the paper, and 0.1 with it, a choice of statsx that comes from
    no paper. The default epochs are the same in both cases: fewer epochs at step size 1 destroy
    local structure without repairing it.

    ``neighbors`` gives the neighbours instead of searching: from an approximate method of
    another library, or from ``nearest_neighbors`` to reuse one search. statsx drops a row's own
    number, computes every distance from ``x``, orders each row by distance then row number, and
    uses the first ``n_neighbors``. The map is then as good as the table: measure it with
    ``fit.neighbor_recall()``. Build the table from the same rows and the same ``metric``.

    ``labels`` lets known classes change the graph, and with it the map. The weight of a
    neighbour with another label is multiplied by ``1 - label_weight``, then each row of weights
    is divided by its largest, so every point keeps one neighbour at weight 1; the two directions
    are then merged as usual. A point labelled -1 has no label: its weights do not change. The
    paper names supervised maps as future work (section 7, one graph per view, intersected) and
    no paper specifies this step: it is a statsx construction. Groups that are apart on such a map
    are not evidence that the data separates them; ``label_agreement`` measures what the data
    supports. ``transform`` takes no labels.

    Args:
        x: The points, shape (n, d), finite.
        n_neighbors: Neighbours per point, from 2 to n - 1. Small values keep fine structure and
            can split the graph into pieces; large values keep the large-scale shape (section
            4.3 of the paper).
        dimensions: The dimension of the map.
        metric: The distance between rows of ``x``. ``"cosine"`` is ``1 - cos`` of the angle, in
            [0, 2]: it ignores the length of a row and is the usual choice for embeddings whose
            direction carries the meaning. The map itself is always Euclidean.
        min_dist: How close two points may sit in the map, in [0, 1]. It changes the look of the
            map, not the graph (section 4.3 of the paper).
        n_epochs: Passes over the edges. 0 is allowed with ``init`` only: the map is the start,
            with the graph of ``x``, so ``cross_entropy`` is the cost of those positions.
        negative_sample_rate: Points drawn at random per sampled edge to push away from.
        init: Start positions, shape (n, dimensions), finite. Row ``i`` is the start of row ``i``
            of ``x``. The default is the spectral start of the paper.
        learning_rate: First step size of the layout, above 0 and at most 1.
        neighbors: Row numbers of neighbours of each row, shape (n, K) with K at least
            ``n_neighbors`` besides the row itself, or ``(indices, distances)``. Given distances
            are only checked against the data: pass Euclidean distances, not squared ones, and
            ``1 - cos`` for the cosine metric.
        labels: The label of each row, shape (n,): integers, 0 or above, and -1 for a row
            without a label.
        label_weight: How much of the weight between different labels is taken away, from 0 (no
            effect) to below 1. Required with ``labels``, with no default: no source gives one.
        seed: Seed of the start and the layout.
        threads: Worker threads. Defaults to the CPUs the process may use, at most 4, or to
            the ``STATSX_THREADS`` environment variable.

    Raises:
        ValueError: If ``n_neighbors`` is not in [2, n - 1], a setting is out of range,
            ``metric`` is ``"cosine"`` and a row is all zeros, or ``init`` has another shape, is
            not finite, or has all its rows in fewer than ``dimensions`` dimensions, or
            ``neighbors`` is not a table of enough different rows with the distances of ``x``, or
            only one of ``labels`` and ``label_weight`` is given, or either is out of range.
    """
    check_metric(metric)
    a = as_points(x)
    n = a.shape[0]
    if not _MIN_NEIGHBORS <= n_neighbors <= n - 1:
        raise ValueError(f"n_neighbors must be in [2, n - 1] = [2, {n - 1}], got {n_neighbors}")
    if dimensions < 1:
        raise ValueError(f"dimensions must be at least 1, got {dimensions}")
    if negative_sample_rate < 0:
        raise ValueError(f"negative_sample_rate must be at least 0, got {negative_sample_rate}")
    epochs = n_epochs if n_epochs is not None else (500 if n <= _LARGE else 200)
    fewest = 1 if init is None else 0
    if epochs < fewest:
        raise ValueError(f"n_epochs must be at least {fewest}, got {epochs}")
    rate = learning_rate
    if rate is None:
        rate = _PAPER_RATE if init is None else _REFRESH_RATE
    if not 0.0 < rate <= _PAPER_RATE:
        raise ValueError(f"learning_rate must be above 0 and at most 1, got {rate}")
    given = None if init is None else _given_start(init, n, dimensions)
    supervised = _supervision_settings(labels, label_weight, n)
    curve_a, curve_b = fit_curve(min_dist)
    workers = resolve_threads(threads)

    if neighbors is None:
        idx, dist = _neighbors(a, n_neighbors, metric, workers)
    else:
        idx, dist = _given_neighbors(a, neighbors, n_neighbors, metric, workers)
    kernel_labels, factor = (
        (None, 1.0) if supervised is None else (supervised[0], 1.0 - supervised[1])
    )
    rho, sigma, indptr, indices, weights, share = _accel.umap_fuzzy(
        idx, dist, kernel_labels, factor
    )
    graph = FuzzyGraph(indptr, indices, weights)
    supervision = (
        None
        if supervised is None
        else Supervision(
            labels=supervised[0],
            label_weight=supervised[1],
            cross_label_share=None if math.isnan(share) else share,
        )
    )
    if given is None:
        start, n_components, n_random, converged = _start(graph, dimensions, seed, workers)
    else:
        start, n_random, converged = given, 0, True
        n_components = int(_accel.umap_components(graph.indptr, graph.indices).max()) + 1
    embedding = (
        start
        if epochs == 0
        else _accel.umap_layout(
            start,
            graph.indptr,
            graph.indices,
            graph.weights,
            curve_a,
            curve_b,
            epochs,
            negative_sample_rate,
            rate,
            seed,
            workers,
        )
    )
    kept = a.copy() if np.may_share_memory(a, x) else a
    kept.flags.writeable = False
    return UMAPFit(
        embedding=embedding,
        graph=graph,
        data=kept,
        neighbors=idx,
        distances=dist,
        metric=metric,
        rho=rho,
        sigma=sigma,
        a=curve_a,
        b=curve_b,
        min_dist=min_dist,
        n_epochs=epochs,
        negative_sample_rate=negative_sample_rate,
        seed=seed,
        n_graph_components=n_components,
        n_random_start=n_random,
        spectral_converged=converged,
        learning_rate=float(rate),
        start="spectral" if given is None else "init",
        neighbor_source="exact" if neighbors is None else "given",
        supervision=supervision,
    )


def label_agreement(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    labels: npt.ArrayLike,
    k: int = 15,
    *,
    query: npt.ArrayLike | None = None,
    query_labels: npt.ArrayLike | None = None,
    metric: Metric = "euclidean",
    threads: int | None = None,
) -> float:
    """Return the mean share of the nearest neighbours of a row that carry its label.

    Without ``query`` each row of ``x`` is compared with its ``k`` nearest other rows of ``x``.
    With ``query`` each row of ``query`` is compared with its ``k`` nearest rows of ``x``: use
    rows that are not in ``x``. 1 means every neighbour has the label of the row; labels that have
    nothing to do with the positions give about the sum of the squared class shares.

    It shows how much of the separation on a map with labels comes from the labels. Compare three
    values: held-out rows against training rows in the data (what the data supports), held-out
    rows placed by ``transform`` against the training positions (what the map kept without their
    labels), and the training positions against themselves (what the picture shows). The measure
    is a choice of statsx and comes from no paper.

    Args:
        x: Reference points, shape (n, d), finite.
        labels: Their labels, shape (n,), integers 0 or above. Leave out rows without a label.
        k: Neighbours per row, from 1 to n - 1, or to n with ``query``.
        query: Rows to score, shape (m, d). The default is ``x`` itself.
        query_labels: Their labels, shape (m,). Required with ``query``.
        metric: The distance between rows.
        threads: Worker threads. The value does not depend on it.

    Raises:
        ValueError: If a shape, a label or ``k`` is wrong, only one of ``query`` and
            ``query_labels`` is given, or ``metric`` is ``"cosine"`` and a row is all zeros.
    """
    check_metric(metric)
    a = as_points(x)
    n = a.shape[0]
    reference = _labels(labels, n, 0, "labels")
    if (query is None) != (query_labels is None):
        raise ValueError("query and query_labels go together")
    most = n - 1 if query is None else n
    if not 1 <= k <= most:
        raise ValueError(f"k must be in [1, {most}], got {k}")
    workers = resolve_threads(threads)
    if query is None or query_labels is None:
        near, _ = _neighbors(a, k, metric, workers)
        own = reference
    else:
        new = as_points(query, min_rows=1)
        if new.shape[1] != a.shape[1]:
            raise ValueError(f"query has {new.shape[1]} columns, x has {a.shape[1]}")
        own = _labels(query_labels, new.shape[0], 0, "query_labels")
        near, _ = _neighbors_between(a, new, k, metric, workers)
    return float((reference[near] == own[:, None]).mean())


def neighbor_preservation(  # noqa: PLR0913 (settings are keyword-only)
    x: npt.ArrayLike,
    embedding: npt.ArrayLike,
    k: int = 15,
    *,
    metric: Metric = "euclidean",
    sample: int | None = None,
    seed: int = 0,
    threads: int | None = None,
) -> float:
    """Return the share of nearest neighbours that the map keeps.

    For each point it takes the ``k`` nearest points in ``x`` and the ``k`` nearest in
    ``embedding``, and the result is the mean share of the first set that is in the second.
    1 means the map keeps every neighbourhood, and a map of unrelated points scores about
    ``k / (n - 1)``. It checks the thing UMAP tries to keep. It says nothing about distances, the
    size of clusters or the gaps between them, which the map does not keep.

    Args:
        x: The original points, shape (n, d).
        embedding: Their positions in the map, shape (n, m).
        k: Neighbours per point, from 1 to n - 1.
        metric: The distance between rows of ``x``. The embedding always uses the Euclidean one.
        sample: Check this many rows, drawn at random, instead of every row: an estimate of the
            same number at a cost of O(sample n d). All rows when there are fewer.
        seed: Seed of the sample.
        threads: Worker threads. The value does not depend on it.
    """
    check_metric(metric)
    original = as_points(x)
    mapped = as_points(embedding)
    n = original.shape[0]
    if mapped.shape[0] != n:
        raise ValueError("x and embedding must have the same number of rows")
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n - 1] = [1, {n - 1}], got {k}")
    workers = resolve_threads(threads)
    rows = None if sample is None else _sample_rows(n, sample, seed)
    if rows is None or rows.shape[0] == n:
        near_x, _ = _neighbors(original, k, metric, workers)
        near_y, _ = _accel.umap_knn(mapped, k, workers)
    else:
        near_x = _neighbors_of(original, rows, k, metric, workers)
        near_y = _neighbors_of(mapped, rows, k, "euclidean", workers)
    shared = (near_x[:, :, None] == near_y[:, None, :]).any(axis=2).sum(axis=1)
    return float(shared.mean()) / k


@dataclass(frozen=True, slots=True)
class PCAFit:
    """A principal component analysis of the rows of a data set.

    Args:
        scores: The coordinates of each row on the components, shape (n, k).
        components: The directions, one per row, shape (k, d), of length 1 and at right angles,
            largest variance first. In each, the entry of largest size is positive.
        mean: The mean of the rows, shape (d,).
        explained_variance: The variance of the rows along each component, divisor n - 1.
        total_variance: The sum of the variances of the d columns, divisor n - 1.
    """

    scores: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    explained_variance: npt.NDArray[np.float64]
    total_variance: float

    @property
    def n_components(self) -> int:
        """The number of components."""
        return int(self.components.shape[0])

    @property
    def explained_variance_ratio(self) -> npt.NDArray[np.float64]:
        """The share of the total variance along each component."""
        return self.explained_variance / self.total_variance

    def transform(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the coordinates of new rows on the components, shape (m, k).

        Raises:
            ValueError: If ``x`` is not a finite array with the columns of the fitted data.
        """
        a = as_points(x, min_rows=1)
        if a.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"x has {a.shape[1]} columns, the fitted data has {self.mean.shape[0]}"
            )
        return _project(a, self.mean, self.components)


def _project(
    a: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    components: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    out = np.empty((a.shape[0], components.shape[0]))
    for start in range(0, a.shape[0], _PCA_BLOCK):
        block = a[start : start + _PCA_BLOCK] - mean
        out[start : start + _PCA_BLOCK] = block @ components.T
    return out


def _directions(
    a: npt.NDArray[np.float64], mean: npt.NDArray[np.float64], k: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the first ``k`` principal directions as rows and the sum of squares along each."""
    n, d = a.shape
    if d <= n:
        # eigenvectors of the d by d scatter matrix, summed over blocks of rows
        scatter = np.zeros((d, d))
        for start in range(0, n, _PCA_BLOCK):
            block = a[start : start + _PCA_BLOCK] - mean
            scatter += block.T @ block
        values, vectors = np.linalg.eigh(scatter)  # ascending
        return vectors[:, ::-1][:, :k].T, np.maximum(values[::-1][:k], 0.0)
    # more columns than rows: the singular vectors of the centred data are the same directions
    _, singular, vt = np.linalg.svd(a - mean, full_matrices=False)
    return vt[:k], singular[:k] ** 2


def pca(x: npt.ArrayLike, n_components: int) -> PCAFit:
    """Return the first ``n_components`` principal components of the rows of ``x``.

    The components are the eigenvectors of the covariance matrix with the largest eigenvalues
    (Pearson 1901; Hotelling 1933). They are exact, not randomized. With d columns and n rows the
    cost is O(n d^2 + d^3) when d <= n, and O(n^2 d) otherwise. The columns are centred and not
    scaled: divide each by its standard deviation first if they have different units.

    The sign of a component is arbitrary in the mathematics. Here the entry of largest size is
    made positive. When two eigenvalues are equal, any pair of directions in their plane is a
    valid answer, and another machine can return another pair.

    NumPy's linear algebra library decides how many threads the products use.

    Args:
        x: The points, shape (n, d), finite.
        n_components: How many components, from 1 to min(n - 1, d).

    Raises:
        ValueError: If ``x`` is not a finite (n, d) array, ``n_components`` is out of range, or
            all rows are equal.
    """
    a = as_points(x)
    n, d = a.shape
    if not 1 <= n_components <= min(n - 1, d):
        raise ValueError(
            f"n_components must be between 1 and min(n - 1, d) = {min(n - 1, d)}, "
            f"got {n_components}"
        )
    mean = a.mean(axis=0)
    total = 0.0
    for start in range(0, n, _PCA_BLOCK):
        block = a[start : start + _PCA_BLOCK] - mean
        total += float(np.einsum("ij,ij->", block, block))
    if total <= 0.0:
        raise ValueError("all rows are equal: there is no direction of variance")
    components, squares = _directions(a, mean, n_components)
    largest = components[np.arange(n_components), np.abs(components).argmax(axis=1)]
    components = np.ascontiguousarray(components * np.where(largest < 0, -1.0, 1.0)[:, None])
    return PCAFit(
        scores=_project(a, mean, components),
        components=components,
        mean=mean,
        explained_variance=squares / (n - 1),
        total_variance=total / (n - 1),
    )
