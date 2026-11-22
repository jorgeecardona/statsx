# Map points: UMAP

Use this when you have many measurements per point and want a map in 2 or 3 dimensions to look at,
or a smaller set of coordinates that keeps each point's neighbours. The function is `umap` in
`statsx.embedding`. It follows McInnes, Healy and Melville (2018).

A map is for looking. It is not evidence of groups. Read [The map is not the
data](#the-map-is-not-the-data) before you cluster or count groups on a map.

```python
import numpy as np
from statsx.embedding import neighbor_preservation, umap

rng = np.random.default_rng(0)
groups = np.vstack([rng.normal(10.0 * c, 1.0, size=(100, 8)) for c in range(3)])
fit = umap(groups)
print(fit.embedding.shape, fit.n_graph_components, fit.spectral_converged)
print(round(neighbor_preservation(groups, fit.embedding, 15), 2))
```

```text
(300, 2) 3 True
0.49
```

`fit.embedding` holds the position of each point. The graph has 3 pieces, one for each group. The
map keeps 49% of each point's 15 nearest neighbours. A map of unrelated points keeps about 5%
(15 of the other 299).

## What it does

UMAP has two phases (section 3 of the paper).

1. Build a graph. Each point is joined to its `n_neighbors` nearest points. The weight of an edge is
   `exp(-max(0, d - rho) / sigma)`, where `d` is the distance, `rho` is the distance to the nearest
   neighbour above zero and `sigma` is solved so that the weights of one point's edges sum to
   `log2(n_neighbors)` (section 3.1). The nearest neighbour always gets weight 1. The two
   directions of an edge are merged as `a + b - a b`, the chance that at least one of them exists.
2. Lay the graph out. The points start at the eigenvectors of the normalized graph Laplacian
   (Algorithm 4). Then each epoch samples every edge with probability equal to its weight and moves
   the point along the gradient of `log Phi`, where `Phi(r) = 1 / (1 + a r^(2b))` is the weight of a
   pair at distance `r` in the map. For each sampled edge it also pushes the point away from
   `negative_sample_rate` points chosen at random, along the gradient of `log(1 - Phi)` (section
   4.1, Algorithm 5). The step size falls from 1 to 0 over the epochs.

`a` and `b` come from `min_dist`: `Phi` is fitted to a curve that is 1 up to `min_dist` and then
falls as `exp(-(r - min_dist))`. A larger `min_dist` spreads points out in the map. It changes the
look of the map and not the graph.

The neighbours are exact. The paper uses an approximate method (nearest-neighbour descent, Dong,
Charikar and Li, 2011); here the cost is O(n^2 d) for the neighbours, which is fine up to some tens
of thousands of points. The layout costs O(`n_neighbors` x n x `n_epochs`). The result does not
depend on `threads`.

## Choose n_neighbors and min_dist

- `n_neighbors` is the size of the area that UMAP treats as flat (section 4.3). A small value keeps
  fine structure and can split the graph into pieces. A large value keeps the large shape and blurs
  detail. The default is 15.
- `min_dist` is in [0, 1]. Use a small value to see dense groups, and a larger one to avoid
  overplotting. The default is 0.1.
- `seed` changes the start of the random sampling. The layout is a search for a minimum, so another
  seed gives another map. Check that what you read from a map holds for two or three seeds.

## Check a map

`neighbor_preservation(x, embedding, k)` is the share of each point's `k` nearest neighbours in `x`
that are still among its `k` nearest in the map. It checks what UMAP tries to keep. Compare it with
a linear map. A swiss roll is a sheet rolled up, and no straight projection can unroll it:

```python
t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=1000))
roll = np.column_stack([t * np.cos(t), 10 * rng.uniform(size=1000), t * np.sin(t)])
fit = umap(roll)
centred = roll - roll.mean(axis=0)
pca = centred @ np.linalg.svd(centred, full_matrices=False)[2][:2].T
kept = round(neighbor_preservation(roll, fit.embedding), 2)
print(kept, round(neighbor_preservation(roll, pca), 2))
print(round(umap(roll, n_epochs=1).cross_entropy()), round(fit.cross_entropy()))
```

```text
0.77 0.39
24901 16200
```

The map keeps 77% of the neighbours, the projection on the two main axes 39%. The second line is
the objective that the layout lowers, the fuzzy set cross entropy of the paper (Definition 10),
summed over all pairs of points. It is 0 when the map reproduces the graph exactly. It was
24,901 after one epoch and 16,200 after the default 500.

Do not compare the entropy of maps of different data or different settings: it has no scale.

## The map is not the data

UMAP keeps neighbours and gives up distances and densities (section 6 of the paper). In a map, the
size of a cluster, the gap between two clusters and where they sit mean nothing.

UMAP also finds structure in noise: the paper says it "can tend to find manifold structure within
the noise of a dataset" (section 6). Here is uniform noise in 10 dimensions. HDBSCAN finds no group
in the data, and finds groups in the map of the same data:

```python
from statsx.clustering import hdbscan

noise = rng.uniform(size=(400, 10))
print(hdbscan(noise, min_cluster_size=15).n_clusters)
print(hdbscan(umap(noise).embedding, min_cluster_size=15).n_clusters)
```

```text
0
5
```

If groups matter, cluster the data, not the map, and use the map to look at the result.

## When the graph falls apart

With a small `n_neighbors`, or with groups that are far apart, the graph can have several connected
pieces. `fit.n_graph_components` counts them. The paper warns that the spectral start needs care in
this case (section 4.3). Here each piece gets the eigenvectors of its own Laplacian, scaled to a box
whose volume follows the number of points, and the boxes sit on a grid. A piece with fewer than
`dimensions + 2` points has no such eigenvectors, and starts at random: `fit.n_random_start` counts
those points. This is a choice of statsx, not a step of the paper.

`fit.spectral_converged` is False when the eigenvector search stopped at 1000 iterations before the
residual fell below 1e-6. The start is then a smooth map of the graph that is close to, but is not,
the eigenvector. The layout still runs.

## What it assumes

- Distance in your space measures similarity, and the Euclidean distance is the right one. Scale
  the columns first if they have different units.
- The data lie near a manifold on which they are spread evenly (the assumptions of section 2 of the
  paper). Where the data are loose in one region and dense in another, UMAP puts both on an even
  footing (section 6).
- There are enough points. The paper warns that results for fewer than 500 points can be poor,
  because of the sampling in the layout (section 6).

## What the library checks and does not check

Checked: the points are a finite (n, d) array, `n_neighbors` is from 2 to n - 1, `min_dist` is in
[0, 1], and the counts are in range.

Reported: the number of graph pieces, the points with a random start, and whether the spectral
start converged.

Not checked: that the layout reached the best map. It is a stochastic search for a local minimum.
Not checked: that the map shows real groups. See above.

## Not included

- Mapping new points into an existing map. The paper lists it as future work (section 7), so
  there is no published algorithm to follow.
- Other distances than the Euclidean one, supervised maps and approximate neighbours.

## Checked in the tests

- The neighbours equal the exact nearest points computed from the full distance matrix, also on an
  integer grid where many distances tie.
- `rho`, `sigma` and the weights equal the formulas of section 3.1, and the graph equals
  `A + A^T - A o A^T`.
- The spectral start equals a dense eigendecomposition: eigenvalues to 1e-9, and the same plane.
- The force on a point equals the gradient of `log Phi` and of `log(1 - Phi)`, by central
  differences. The repulsive force differs from the exact gradient by the relative `eps / r^2` of
  the paper.
- The cross entropy equals Definition 10 summed over all pairs, written out in the test.
- The layout reaches the cross entropy of Algorithm 5 run one point at a time, within 15%, on
  random starts. The two differ in one way: a point reads the other end of an edge at its position
  at the start of the epoch.
- The fitted `a` and `b` for `min_dist = 0.1` are 1.577 and 0.895 (the documentation of uwot lists
  1.58 and 0.90), and are a least squares minimum.
- On a swiss roll the map keeps 60% or more of the neighbours, and 20 points more than the two
  main axes do.
- Three groups far apart stay apart. Uniform noise in 10 dimensions gave 0 clusters in the data and
  2 or more in the map, for 4 of 4 seeds.
- The map is identical with 1 and with 3 threads, bit for bit.
- 20,000 points in 20 dimensions, in 10 groups, took 6.7 s with 4 threads (200 epochs), on a machine
  in use.

## References

- McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction. arXiv:1802.03426 (version 3, 2020).
  <https://arxiv.org/abs/1802.03426>
- Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver: locally optimal block
  preconditioned conjugate gradient method. *SIAM Journal on Scientific Computing* 23(2):517-541.
  <https://doi.org/10.1137/S1064827500366124>
- Duersch, J. A., Shao, M., Yang, C., and Gu, M. (2018). A robust and efficient implementation of
  LOBPCG. *SIAM Journal on Scientific Computing* 40(5):C655-C676.
  <https://doi.org/10.1137/17M1129830>
- Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares.
  *Quarterly of Applied Mathematics* 2(2):164-168.
- Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters.
  *Journal of the Society for Industrial and Applied Mathematics* 11(2):431-441.
- Tang, J., Liu, J., Zhang, M., and Mei, Q. (2016). Visualizing large-scale and high-dimensional
  data. *Proceedings of the 25th International Conference on World Wide Web*, pp. 287-297.
  <https://doi.org/10.1145/2872427.2883041>
- Dong, W., Charikar, M., and Li, K. (2011). Efficient k-nearest neighbor graph construction for
  generic similarity measures. *Proceedings of the 20th International Conference on World Wide
  Web*, pp. 577-586. <https://doi.org/10.1145/1963405.1963487>
