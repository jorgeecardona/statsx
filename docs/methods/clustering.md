# Cluster points: k-means, the silhouette and how many clusters

Use this when you have points with no labels (measurements, embeddings) and want to split them into
groups, check the groups, and choose how many there are. The module is `statsx.clustering`.

## k-means

`kmeans(x, k)` finds `k` centroids that minimize the within-cluster sum of squares

\[
W = \sum_{c=1}^{k} \sum_{i \in c} \lVert x_i - \mu_c \rVert^2 .
\]

Each run seeds the centroids by k-means++ (Arthur and Vassilvitskii 2007), then repeats Lloyd's two
steps (Lloyd 1982): assign each point to its nearest centroid, then move each centroid to the mean of
its points. `n_init` runs from different seeds are made and the run with the lowest `W` is kept,
because one run can stop in a poor local minimum. The same `seed` gives the same fit for any number of
threads.

```python
import numpy as np
from statsx.clustering import choose_k, kmeans, silhouette

rng = np.random.default_rng(0)
centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
x = np.vstack([c + 0.7 * rng.standard_normal((60, 2)) for c in centres])
fit = kmeans(x, 3, seed=1)
print(fit.sizes, round(fit.inertia, 1), fit.converged)
print((np.round(fit.centroids, 1) + 0.0).tolist())
```

```text
[60 60 60] 176.1 True
[[0.0, 7.9], [7.8, 0.0], [0.0, 0.1]]
```

`fit.predict(new_points)` gives the nearest centroid of new rows, and `fit.distances(new_points)` also
gives the squared distance to it. A large distance says the point is far from every cluster.

## The silhouette

`silhouette(x, labels)` shows which points sit between clusters (Rousseeuw 1987). For a point, `a` is its
mean distance to the other points of its cluster and `b` the smallest mean distance to another
cluster. The silhouette is `(b - a) / max(a, b)`, from -1 to 1. Near 1 the point is well inside its
cluster. Near 0 it is on a border. Below 0 it is nearer another cluster.

```python
s = silhouette(x, fit.labels)
print(round(s.mean, 2), s.n_negative, np.round(s.by_cluster, 2).tolist())
```

```text
0.84 0 [0.84, 0.83, 0.84]
```

The cost grows with the square of the number of rows. Use a sample for more than about 20,000 rows.

## How many clusters

`choose_k(x, k_max)` computes the gap statistic (Tibshirani, Walther and Hastie 2001). For each `k`
it compares `log W_k` of the data with the same quantity on uniform reference data in the range of
the data. A large gap means the data are tighter at `k` than uniform data are. It picks the smallest
`k` with `gap[k] >= gap[k + 1] - se[k + 1]`.

```python
gap = choose_k(x, 6, seed=0)
print(gap.k, gap.found, np.round(gap.gap, 2).tolist())
```

```text
3 True [-0.21, -0.06, 2.04, 1.87, 1.73, 1.65]
```

A pick of `k = 1` says the data are not more clustered than uniform data. It does not say they are one
round cluster. If no `k` meets the rule, `found` is False and the pick is the largest `k` tried.

## What it assumes

- k-means assumes round clusters of similar spread and size. It minimizes a sum of squares, which is
  the maximum likelihood partition under those conditions.
- Distances mean something in your space. Euclidean distance on unscaled columns is dominated by the
  column with the largest range. For text embeddings, divide each row by its norm first.
- The gap statistic assumes the null model "uniform over the range of the data". Data that are not
  uniform and not clustered, such as one long skewed cloud, may get a pick above 1.

## What the library checks and does not check

Checked: the points are a finite (n, d) array with at least 2 rows, `k` is from 1 to n, and the
counts and tolerances are in range. A fit reports `converged`, and says False when a run reached
`max_iter`.

Not checked: that the clusters exist. k-means returns `k` clusters for any data, including one
Gaussian cloud, so a result is no evidence of structure. Read the silhouette and the gap. Neither
proves that the clusters are real: they say how well separated the partition is.

## Checked in the tests

- The fit reaches the best partition, found by trying all assignments, on 10 points with 3 clusters.
- The result is identical with 1 and with 4 threads, bit for bit.
- The silhouette equals its definition computed by hand on random data.
- Three planted blobs gave `k = 3` in 40 of 40 runs of `choose_k`. Uniform data gave `k = 1` in 38 of
  40 runs, and `k = 2` in the others.
- 200,000 points in 32 dimensions with `k = 20` took 4.8 s for 300 iterations on 4 threads.

## References

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*
  28(2):129-137. <https://doi.org/10.1109/TIT.1982.1056489>
- Arthur, D., and Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding.
  *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, pp. 1027-1035.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of
  cluster analysis. *Journal of Computational and Applied Mathematics* 20:53-65.
  <https://doi.org/10.1016/0377-0427(87)90125-7>
- Tibshirani, R., Walther, G., and Hastie, T. (2001). Estimating the number of clusters in a data set
  via the gap statistic. *Journal of the Royal Statistical Society B* 63(2):411-423.
  <https://doi.org/10.1111/1467-9868.00293>
