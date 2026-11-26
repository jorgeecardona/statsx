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
kept = neighbor_preservation(groups, fit.embedding, 15)
print(kept > 0.3)
```

```text
(300, 2) 3 True
True
```

`fit.embedding` holds the position of each point. The graph has 3 pieces, one for each group. The
map keeps more than 30% of each point's 15 nearest neighbours (49% in the run that wrote this
page). A map of unrelated points keeps about 5% (15 of the other 299).

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
  seed gives another map. The same seed can also give a slightly different map on another machine,
because the layout is chaotic and the last digits of a distance differ. Check that what you read
from a map holds for two or three seeds.

## Use the cosine distance

Use `metric="cosine"` when the direction of a row carries the meaning and its length does not, as
with text embeddings. The distance is `1 - cos` of the angle between two rows, so scaling a row
changes nothing. The map itself is still Euclidean.

```python
lengths = np.exp(rng.uniform(0.0, np.log(100.0), size=(200, 1)))
rays = np.vstack([np.ones(10), np.r_[np.ones(5), -np.ones(5)]])
data = (rays[np.repeat([0, 1], 100)] + rng.normal(0.0, 0.2, size=(200, 10))) * lengths
fit = umap(data, metric="cosine")
same = np.repeat([0, 1], 100)[fit.neighbors] == np.repeat([0, 1], 100)[:, None]
print(fit.metric, same.mean() > 0.99, fit.distances.max() <= 2.0)
```

```text
cosine True True
```

Each point's 15 neighbours come from its own ray, whatever the length. `fit.distances` holds the
cosine distances, in [0, 2]. A row of zeros has no direction, so the function refuses it.

Two rows in the same direction are at distance 0, and the graph then treats them as the paper treats
duplicates: `rho` is the smallest distance above zero. The function counts rows as one direction
when their unit vectors are within 1e-14, which is rounding.

The cosine distance is not a metric: it breaks the triangle inequality. The paper builds the graph
from the distances to the neighbours of each point and nothing else (section 3.1), so any
dissimilarity can feed it. The paper states its theory for metric spaces, so a cosine map goes
beyond what the paper proves.

## Check a map

`neighbor_preservation(x, embedding, k)` is the share of each point's `k` nearest neighbours in `x`
that are still among its `k` nearest in the map. Pass `metric="cosine"` if you mapped with it. It
checks what UMAP tries to keep. Compare it with a linear map. A swiss roll is a sheet rolled up,
and no straight projection can unroll it:

```python
t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=1000))
roll = np.column_stack([t * np.cos(t), 10 * rng.uniform(size=1000), t * np.sin(t)])
fit = umap(roll)
centred = roll - roll.mean(axis=0)
pca = centred @ np.linalg.svd(centred, full_matrices=False)[2][:2].T
kept = neighbor_preservation(roll, fit.embedding)
print(kept > 0.6, neighbor_preservation(roll, pca) < 0.5)
print(fit.cross_entropy() < 0.8 * umap(roll, n_epochs=1).cross_entropy())
```

```text
True True
True
```

The first line says that the map keeps more than 60% of the neighbours and the projection on the
two main axes keeps less than 50%. In the run that wrote this page they were 77% and 39%. The second
line compares the objective that the layout lowers, the fuzzy set cross entropy of the paper
(Definition 10), summed over all pairs of points. It is 0 when the map reproduces the graph exactly.
Here the default 500 epochs lowered it by more than a fifth against one epoch (24,901 to 16,200).

Do not compare the entropy of maps of different data or different settings: it has no scale.

## Place new points in a map

Use `fit.transform(new_points)` when the map is finished and new rows arrive. The map does not
change. Each new point is joined to its `n_neighbors` nearest training points, starts at the
weighted mean of their positions, and is then moved by the same sampled gradient steps as the fit,
with the training points held fixed. The result is a `UMAPPlacement`.

The weights are the ones of the fit with one change: `rho` is 0, so a weight is `exp(-d / sigma)`
and the nearest training point does not get weight 1 by force. The uwot article and the umap-learn
code do the same for new points.

```python
train, new = roll[:800], roll[800:]
fit = umap(train)
placed = fit.transform(new)
print(placed.embedding.shape, placed.neighbors.shape)
print(placed.distances[:, 0].max() < 3 * fit.distances[:, 0].max())
```

```text
(200, 2) (200, 15)
True
```

`placed.distances[:, 0]` is how far each new point is from the training data. Compare it with
`fit.distances[:, 0]`. A new point much farther out than the training points lands where the map
has no information, so do not read its place as a result.

`n_epochs=0` returns the weighted mean of the neighbours and does no gradient step. The default is
a third of the epochs of the fit, at least 30. The result does not depend on `threads`. The
random stream of a point is seeded by its row number in the call.

This step is not in the paper. The paper lists it as future work (section 7), and Reichmann,
Hägele and Weiskopf (2024) state that no paper describes it. The method here is the one that the
uwot article "Embedding New Data" describes and umap-learn follows, written again from that
description. Treat it as an implementation choice and not as a published algorithm.

## Refresh a map

Use `umap(x, init=positions)` when the data change over time and the new map has to stay
comparable with the old one. Without it, a new map of nearly the same rows can come out turned,
mirrored or arranged differently.

`init` has one start position per row of `x`, shape (n, dimensions). The simplest start is the old
map's own placement of every row. Here 200 rows leave a window and 200 new rows enter it:

```python
before = umap(roll[:800])
window = roll[200:]
after = umap(window, init=before.transform(window).embedding)
cold = umap(window)


def movement(fit_of_window):
    shared = fit_of_window.embedding[:600] - before.embedding[200:]
    return np.median(np.linalg.norm(shared, axis=1))


print(after.start, after.learning_rate)
print(movement(after) < 0.3 * movement(cold))
```

```text
init 0.1
True
```

If you know which rows stayed, you can give them their old positions and place only the new ones:

```python
start = np.vstack([before.embedding[200:], before.transform(roll[800:]).embedding])
after = umap(window, init=start)
print(movement(after) < 0.3 * movement(cold))
```

```text
True
```

### What happens to `init`

- It is used as given: no rescaling and no noise. Row `i` starts where you put it.
- It is copied. The layout never writes your array.
- `n_epochs=0` returns the start with the graph of the new data. `cross_entropy()` of that fit is
  the cost of the old positions on the new data.
- It is refused when its rows all lie on one point, line or plane of a lower dimension than the
  map. The layout could not leave it.

### The step size

`learning_rate` is the first step size of the layout (Algorithm 5), which falls linearly to 0. The
default is 1.0 without `init`, the value of the paper, and 0.1 with `init`. The value 0.1 is a
choice of statsx and comes from no paper.

Measured on 600 points in 8 Gaussian groups of 64 columns, 3 data sets. 100 rows left, 100 new
rows from the same groups entered, and every new row started at `before.transform`. The map is 58
to 60 wide. "Moved" is the median distance between the old and new position of the 500 rows that
stayed. "Kept" is `neighbor_preservation` with 15 neighbours.

| Run | Moved | Kept | Cross entropy against a cold fit |
|---|---|---|---|
| cold fit | not comparable | 0.354 to 0.358 | 1.00 |
| `learning_rate=1.0`, 500 epochs | 2.5 to 5.3 | 0.352 to 0.356 | 0.996 to 1.010 |
| `learning_rate=0.1`, 500 epochs (default) | 1.1 to 1.4 | 0.385 to 0.389 | 0.961 to 0.977 |
| `learning_rate=0.1`, 200 epochs | 0.8 | 0.376 to 0.379 | 0.974 to 0.982 |

Fewer epochs at step size 1 are not a substitute. On the same kind of data with unchanged rows,
50 epochs kept 0.29 to 0.30 of the neighbours, less than the 0.35 of the old positions.

This is one shape of data. Check your own maps with `neighbor_preservation` and `cross_entropy`
against a cold fit of the same rows. Both fits have the same graph, so the two values compare.

### What a refresh does not promise

- A row whose data did not change does not stay exactly where it was. The example below the list
  measures how far rows move with no change at all.
- A row whose data changed may not reach its new place if its start is its old position and
  `learning_rate` is below 1. In one test, rows moved to another group travelled 7 to 9 of the 34
  to 40 units they needed. Place such rows with `transform`, or pass `learning_rate=1.0`.
- A group of rows that the old map never saw has no right place in it. `transform` reports how far
  each row is from the old data in `placement.distances`.
- A chain of refreshes adds the movement of each step. Compare neighbouring maps, or start every
  refresh from one reference map.
- A start from another source (PCA scores, another library's map) was not measured. Use
  `learning_rate=1.0` for it.

To read movement as change, compare it with a refresh of the old data, where nothing changed:

```python
baseline = umap(before.data, init=before.embedding)
steps = np.linalg.norm(baseline.embedding - before.embedding, axis=1)
print(np.median(steps) > 0.0)
```

```text
True
```

## Save and load a fit

Use `fit.to_npz(path)` to keep a fit, and `UMAPFit.from_npz(path)` to read it back. The file holds
the map, the graph, the fitted data and the settings. A loaded fit places new points exactly as
the saved one does.

```python
import io

from statsx.embedding import UMAPFit

buffer = io.BytesIO()
fit.to_npz(buffer)
buffer.seek(0)
loaded = UMAPFit.from_npz(buffer)
same = np.array_equal(loaded.transform(new[:20]).embedding, fit.transform(new[:20]).embedding)
print(same, loaded.n_points == fit.n_points)
```

```text
True True
```

The file is a NumPy `.npz` with no pickled objects, so loading it runs no code. Every value is
stored exactly. It keeps the fitted data, so it is as large as the data. A path is written as you
give it, and no `.npz` is added.

The format has version 2. `from_npz` also reads version 1, written by statsx 0.1.79 and before.

`from_npz` checks the file before it uses it. It refuses an unknown format version, a missing array,
shapes that do not fit together, a neighbour or a graph entry that points to a row that does not
exist, and values that are not finite. It raises `ValueError` in each case. It does not check that
the map is a good map of the data: a file that someone edited with consistent values loads.

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
print(hdbscan(umap(noise).embedding, min_cluster_size=15).n_clusters >= 2)
```

```text
0
True
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

- Distance in your space measures similarity, and the Euclidean or the cosine distance is the
  right one. Scale the columns first if they have different units and you use the Euclidean one.
- The data lie near a manifold on which they are spread evenly (the assumptions of section 2 of the
  paper). Where the data are loose in one region and dense in another, UMAP puts both on an even
  footing (section 6).
- There are enough points. The paper warns that results for fewer than 500 points can be poor,
  because of the sampling in the layout (section 6).

## What the library checks and does not check

Checked: the points are a finite (n, d) array, `metric` is known and no row is zero for the cosine
distance, `n_neighbors` is from 2 to n - 1, `min_dist` is in
[0, 1], and the counts are in range.

Reported: the number of graph pieces, the points with a random start, whether the spectral
start converged, where the start came from (`fit.start`) and `fit.learning_rate`.

Not checked: that the layout reached the best map. It is a stochastic search for a local minimum.
Not checked: that the map shows real groups. See above.

## Not included

- A published algorithm for mapping new points. There is none (section 7 of the paper), so
  `transform` follows the description in the uwot article on embedding new data.
- Other distances than the Euclidean and cosine ones, supervised maps and approximate neighbours.

## Checked in the tests

- The neighbours equal the exact nearest points computed from the full distance matrix, also on an
  integer grid where many distances tie.
- `rho`, `sigma` and the weights equal the formulas of section 3.1, and the graph equals
  `A + A^T - A o A^T`.
- The spectral start equals a dense eigendecomposition: eigenvalues to 1e-9, and the same plane.
- The force on a point equals the gradient of `log Phi` and of `log(1 - Phi)`, by central
  differences. The repulsive force differs from the exact gradient by the relative `eps / r^2` of
  the paper.
- With `metric="cosine"` the neighbours and distances equal the exact `1 - cos` from the full
  matrix, scaling a row changes nothing, a scaled copy of a row is at distance 0, and groups that
  differ only in direction keep their neighbours when the lengths differ a hundredfold.
- A saved fit loads back with every array and setting equal, also for a seed above 2^63, and places
  new points exactly as before. A file of an unknown version, with a missing array, with a neighbour
  or a graph entry out of range, with a NaN, or with a pickled object is refused.
- `transform` finds the exact nearest training points, weights them with `rho` 0 so that the
  weights sum to `log2(k)`, and with 0 epochs it returns their weighted mean. On a swiss roll the
  new points keep about as many neighbours as the training points do (0.72 against 0.77, with 1000
  training points and 250 new ones). A training point placed again lands within 0.6 of its
  nearest-neighbour distance from where it was. The result is identical with 1 and with 3
  threads.
- With `init`: 0 epochs return the start in a new array, the caller's array is unchanged also when
  it is read-only, a start turned by 90 degrees gives a turned map, rows with one start position
  come apart, the result is identical with 1 and with 3 threads, and a moving window keeps as many
  neighbours (within 0.03) as a cold fit while its shared rows move under 0.3 as far.
- A saved version 1 file loads with `learning_rate` 1.0 and a spectral start.
- The cross entropy equals Definition 10 summed over all pairs, written out in the test.
- The layout reaches the cross entropy of Algorithm 5 run one point at a time, within 30%, on
  random starts (the ratio was 0.84 to 1.10 on two machines). The two differ in one way: a point
  reads the other end of an edge at its position at the start of the epoch.
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
  <https://doi.org/10.1145/2872427.2883041>. It writes the linear schedule of the step size with
  its first value as a parameter.
- Kobak, D., and Linderman, G. C. (2021). Initialization is critical for preserving global data
  structure in both t-SNE and UMAP. *Nature Biotechnology* 39:156-157.
  <https://doi.org/10.1038/s41587-020-00809-z>
- Dong, W., Charikar, M., and Li, K. (2011). Efficient k-nearest neighbor graph construction for
  generic similarity measures. *Proceedings of the 20th International Conference on World Wide
  Web*, pp. 577-586. <https://doi.org/10.1145/1963405.1963487>
- Reichmann, L., Hägele, D., and Weiskopf, D. (2024). Out-of-core dimensionality reduction for
  large data via out-of-sample extensions. arXiv:2408.04129. It states that no paper describes the
  out-of-sample extension of UMAP (section 3.2.1). <https://arxiv.org/abs/2408.04129>
- uwot documentation. Embedding New Data. Article of the R package uwot (version 0.2.5.9000).
  <https://jlmelville.github.io/uwot/articles/embedding-new-data.html>
- umap-learn documentation. Transforming New Data with UMAP, and the `transform` method in
  `umap/umap_.py`, which lowers the local connectivity by one for new points.
  <https://umap-learn.readthedocs.io/en/latest/transform.html>
