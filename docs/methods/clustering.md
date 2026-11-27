# Cluster points: k-means, Gaussian mixtures, DBSCAN and HDBSCAN

Use this when you have points with no labels (measurements, embeddings) and want to split them into
groups, check the groups, and choose how many there are. The module is `statsx.clustering`.

Use k-means for round groups of similar size. Use a Gaussian mixture when groups are stretched,
differ in spread, or overlap, or when you want a probability for each point. Use DBSCAN or HDBSCAN
when groups have any shape, you do not know how many there are, and some points belong to no group.

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

A row joins the centroid with the smallest distance computed in float64, and the first of two at the
same distance. With more than 8 centroids, a matrix product first rules out those that its proven
error bound leaves no chance (Higham 2002, section 3.1). The product only saves time: the labels are
the same on any processor.

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

### float32 data

`kmeans`, `assign_clusters`, `predict`, `assign` and `distances` read a C-contiguous float32 array
where it is, without a float64 copy, and return the same results as for that copy. Embeddings stored
in float32 then need half the memory. The results are float64.

`metric="cosine"` still makes a float64 copy of the rows, scaled to length 1. So do the other
functions of this page.

## Many rows: mini-batch k-means

Use `minibatch_kmeans(x, k)` when `kmeans` is too slow for your rows. It is an approximation:
compare the two on a sample first. It is Algorithm 1 of Sculley (2010).

```python
from statsx.clustering import minibatch_kmeans

many = np.vstack([c + 0.7 * rng.standard_normal((20_000, 2)) for c in centres])
full = kmeans(many, 3, n_init=1)
part = minibatch_kmeans(many, 3, batch_size=1000)
print(part.method, part.converged, part.inertia < 1.01 * full.inertia)
```

```text
minibatch False True
```

What it does:

1. Starts the centres by k-means++ on a first random sample of `max(batch_size, 3 k)` rows. The
   paper starts from random rows.
2. `n_batches` times: draws `batch_size` rows with replacement, finds the nearest centre of each,
   then moves that centre towards the row by 1 over the number of rows it has had.
3. Gives every row its nearest centre.

The result is a `KMeansFit` with `method="minibatch"`. Its centres are not the means of their rows,
and nothing tests that it converged. `predict`, `assign` and `dispersion` work as for `kmeans`.
There is no cosine distance here.

The defaults, 10,000 rows and 30 batches, are choices of `statsx`. On 100,000 rows of 1,024
columns in 8 groups, with `k = 40` on 4 threads, the defaults took 5 s and ended with an inertia
0.05% above `kmeans` with `n_init=1`, which took 52 s (on a busier machine). 100 batches of 1,000
rows took 2 s and ended 0.18% above. On 6 data sets of 64 columns in 40 groups, the defaults ended
on average 1.3% above the best of three `kmeans` runs, and 5.5% above in the worst of 18 runs.

## Size and spread of each cluster

`fit.sizes` counts the rows of each cluster. `fit.dispersion` is the mean distance of those rows to
their centroid: Euclidean (not squared), or `1 - cos`. A wide cluster and a tight one of the same
size differ here. A wide cluster is a candidate to split, and is not proof that it holds two
groups.

```python
uneven = np.vstack([rng.normal(0.0, 0.1, size=(200, 4)), rng.normal(20.0, 3.0, size=(200, 4))])
wide = kmeans(uneven, 2)
print(wide.sizes.tolist(), wide.dispersion.max() > 10 * wide.dispersion.min())
```

```text
[200, 200] True
```

## Assign new rows and mark the far ones

Use `fit.assign(new_rows, max_distance=...)` when rows that arrive later may belong to no cluster.
Rows farther than the limit get the label -1. `assign_clusters(rows, centroids)` does the same for
centres of your own, with no fit.

```python
from statsx.clustering import assign_clusters

limit = float(np.quantile(fit.assign(x).distances, 0.99))
new = np.array([[0.5, 0.2], [7.5, 0.4], [30.0, 30.0]])
found = fit.assign(new, max_distance=limit)
print((found.labels >= 0).tolist(), found.n_noise, found.distances[2] > limit)
same = assign_clusters(new, fit.centroids, max_distance=limit)
print(np.array_equal(same.labels, found.labels))
```

```text
[True, True, False] 1 True
True
```

The distances of `assign` are Euclidean (not squared), or `1 - cos` for a cosine fit. They differ
from `fit.distances`, which returns the squared distance. `found.nearest` keeps the nearest centre
of the rows labelled -1.

A limit is a choice you make. The quantile above says "farther than 99% of the rows of the fit",
so about 1 row in 100 of the same kind is marked too.

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

### A pick that is decided by noise

Read `gap.decision_margin` before you use `gap.k`. It says how far the closest comparison on the way
to the pick was from going the other way, in standard errors. Below 1, treat the pick as
unsettled. `gap.margins` holds `gap[k] - (gap[k+1] - se[k+1])` for each `k`.

In many dimensions the gap curve can be flat: `log W_k` falls at nearly the same rate for the data
and for the reference. Here are 8 groups of directions in 128 dimensions:

```python
draw = np.random.default_rng(0)
centres = draw.normal(size=(8, 128))
centres /= np.linalg.norm(centres, axis=1, keepdims=True)
flat = centres[draw.integers(0, 8, size=600)] + 0.35 * draw.normal(size=(600, 128))
flat /= np.linalg.norm(flat, axis=1, keepdims=True)
runs = [choose_k(flat, 6, n_reference=b, n_init=1, reference="box") for b in (2, 5, 10)]
print(all(r.found for r in runs), all(r.decision_margin < 1 for r in runs))
```

```text
True True
```

`found` is True for each run, and each says that its pick is unsettled. More references do not make
`se` smaller: it is the spread of the reference `log W_k`, and with 2 or 3 references that spread
is poorly estimated. When the margin is small, use `elbow_k` or decide on other grounds.

A margin above 1 says that the pick is stable. It does not say that it is right.

On data with many columns, reduce them with [PCA](pca.md) before you cluster: k-means can miss
groups in hundreds of columns that it finds on the first components.

`k_min` starts the rule at a larger `k`, if one cluster is no answer for you. The result can then
never say 1.

A call runs `(k_max - k_min + 1) * (n_reference + 1) * n_init` k-means fits, 1050 with the defaults.
Lower `n_init` or `k_max` before you lower `n_reference` below 10.

## The elbow of the inertia curve

Use `elbow_k(x, k_max)` when you want the number of clusters where more clusters stop making them
much tighter. It runs k-means for each `k` from `k_min` (2) to `k_max`, and applies the Kneedle
algorithm (Satopaa, Albrecht, Irwin and Raghavan 2011) to the logarithm of the inertia.

```python
from statsx.clustering import elbow_k

elbow = elbow_k(x, 8, seed=0)
print(elbow.k, elbow.is_knee, elbow.inertia_rises)
```

```text
3 True 0
```

The curve is scaled to the unit square and turned so that it rises. The `difference` is how far it
is above the straight line from the first candidate to the last. `k` is the candidate with the
largest difference. `is_knee` says that the algorithm's threshold declares a knee there: after the
peak, the difference falls by more than `sensitivity / (number of candidates - 1)` before it rises
again.

An elbow cannot tell you that there are no clusters. It never answers 1, and the curve of uniform
data bends too. Here the elbow picks a `k` for uniform noise, and the gap statistic answers 1:

```python
flat = np.random.default_rng(0).uniform(size=(500, 5))
print(elbow_k(flat, 10).k > 1, choose_k(flat, 6).k)
```

```text
True 1
```

Read `inertia_rises`. The best possible inertia never rises with `k`, so a value above 0 means
that a fit missed it: raise `n_init`. `scale="linear"` uses the inertia itself. The logarithm is
the default because large early drops can hide the bend on the raw values; it is a choice taken
from fastumap, not part of the paper. The paper also smooths the data with a spline. `elbow_k` does
not smooth: it has one value per `k`.

`find_knees(x, y)` is the algorithm alone, for any curve that rises and bends downwards.

## The cosine distance

Pass `metric="cosine"` when the direction of a row carries the meaning and its length does not, as
with text embeddings. The distance is `1 - cos` of the angle between two rows. `kmeans`,
`elbow_k`, `dbscan`, `k_distances` and `hdbscan` take it. A row of zeros has no direction, and the
functions refuse it.

```python
from statsx.clustering import dbscan, elbow_k, hdbscan, k_distances

axes = np.eye(6)[:3]
rays = np.vstack([a + rng.normal(0.0, 0.12, size=(80, 6)) for a in axes])
rays *= np.exp(rng.uniform(np.log(0.1), np.log(100.0), size=(240, 1)))
fit = kmeans(rays, 3, metric="cosine")
print(fit.sizes.tolist(), elbow_k(rays, 9, metric="cosine").k)
eps = float(np.quantile(k_distances(rays, 5, metric="cosine"), 0.95))
print(dbscan(rays, eps, metric="cosine").n_clusters)
print(hdbscan(rays, 15, metric="cosine").n_clusters)
```

```text
[80, 80, 80] 3
3
3
```

The rows have lengths from 0.1 to 100, and the three groups are found by direction.

For k-means this is spherical k-means (Dhillon and Modha 2001): each centre has length 1, a row
joins the centre with the largest cosine, and `inertia` is the sum of `1 - cos`. `fit.distances`
returns `1 - cos` to the centre. Scaling the rows yourself and calling `kmeans` without a metric
is not the same: a Euclidean centre keeps its length, which is lower for a wider cluster, and the
labels can differ.

For DBSCAN, `eps` and `k_distances` are in `1 - cos`, from 0 to 2. For HDBSCAN the neighbours and
the tree are those of the unit rows, and the densities are `1 / (1 - cos)`. So HDBSCAN on rows
you scaled yourself can choose other clusters.

`1 - cos` is not a metric: it breaks the triangle inequality. DBSCAN and HDBSCAN are defined for
any distance between two points, and k-means with it is the objective of Dhillon and Modha.

`choose_k` (the gap statistic) does not take a metric. Its reference data are uniform in a box,
which says nothing about directions. `silhouette` uses the Euclidean distance.

## Gaussian mixtures

`gaussian_mixture(x, k)` fits a mixture of `k` Gaussians by expectation maximization, EM (Dempster,
Laird and Rubin 1977). The density of a point is

\[
p(x) = \sum_{j=1}^{k} \pi_j \, N(x \mid \mu_j, \Sigma_j) .
\]

Each EM iteration has two steps. The E-step gives each point a probability for each component (its
responsibility). The M-step sets each weight, mean and covariance to the responsibility-weighted
values. The log-likelihood does not decrease at any iteration, and `fit.trace` shows it.

The first run starts from a converged k-means partition. The other runs start from k-means++ seeds
after 0 or 3 Lloyd steps. `n_init` runs are made (10 by default) and the one with the highest
log-likelihood is kept. The same `seed` gives the same fit for any number of threads.

EM stops at the optimum nearest its start, so one run can end far from the best. On the data below,
a single run from the k-means partition reaches a log-likelihood about 430 below the best. Ten
runs reached the best on every one of 10 seeds. More runs make a poor optimum less likely. They do
not rule it out.

```python
from statsx.clustering import choose_components, gaussian_mixture

rng = np.random.default_rng(1)
long_axis = np.array([[3.0, 0.0], [0.0, 0.3]])
a = rng.standard_normal((400, 2)) @ long_axis
b = rng.standard_normal((400, 2)) @ long_axis + np.array([0.0, 2.0])
c = rng.standard_normal((200, 2)) * 0.5 + np.array([7.0, 7.0])
y = np.vstack([a, b, c])
truth = np.repeat([0, 1, 2], [400, 400, 200])

mix = gaussian_mixture(y, 3, seed=1)
print(np.round(np.sort(mix.weights), 2).tolist(), mix.converged, mix.n_iter)
print(np.round(mix.means[np.argsort(mix.weights)], 1).tolist())
```

```text
[0.2, 0.4, 0.4] True 113
[[7.0, 7.0], [-0.4, -0.0], [0.2, 2.0]]
```

Two stretched groups lie close together. k-means cuts each group by distance, so it mixes them. The
mixture uses the covariance, so it separates them.

```python
def agreement(labels):
    """Share of points whose group matches the planted one, over the best relabelling."""
    import itertools
    return max(
        np.mean(np.array(p)[labels] == truth) for p in itertools.permutations(range(3))
    )

print(round(float(agreement(kmeans(y, 3, seed=1).labels)), 2))
print(round(float(agreement(mix.labels)), 2))
```

```text
0.62
1.0
```

### Probabilities and new points

`mix.responsibilities` has one row per point and one column per component, and each row sums to 1.
`mix.uncertainty` is `1 - max responsibility`: 0 is sure, and a value near 0.5 is a toss between two
components. `mix.predict_proba(new_points)` scores new rows. `mix.predict(new_points)` gives the
most probable component. `mix.score_samples(new_points)` gives the log density. A very low log
density says the point is far from every component.

```python
print(round(float(mix.uncertainty.mean()), 3), int((mix.uncertainty > 0.25).sum()))
new = np.array([[0.0, 1.0], [7.0, 7.0], [-30.0, 30.0]])
print(np.round(mix.predict_proba(new), 2).tolist())
print(np.round(mix.score_samples(new), 1).tolist())
```

```text
0.0 0
[[0.0, 0.66, 0.34], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
[-7.3, -2.0, -3972.1]
```

### Covariance forms

`covariance="full"` lets each component be any ellipse and costs d(d+1)/2 numbers per component.
`"diag"` keeps the axes along the columns. `"spherical"` uses one variance per component, and with
equal weights and equal variances it is k-means with soft labels. Fewer numbers need fewer points
and are steadier in high dimensions.

### Components that collapse

A component can shrink onto a few points. Its variance goes to 0 and the likelihood grows without
bound (McLachlan and Peel 2000). That is a fit to noise, not a cluster. A run is dropped when a
component holds fewer than d + 1 points in effect (`"full"`) or 2 points (`"diag"`, `"spherical"`).
`fit.n_degenerate` counts dropped runs. If every run is dropped, the call raises `ValueError`: use
fewer components, a simpler covariance, or more data. `reg` adds a small share of the mean variance
to every variance, so a covariance stays positive definite.

## How many components

`choose_components(x, k_max)` fits `k = 1 .. k_max` and picks the lowest BIC (Schwarz 1978):

\[
\mathrm{BIC} = -2 \log L + p \log n ,
\]

where `p` counts weights, means and covariances. AIC (`criterion="aic"`) uses `2p` and asks for more
components. A `k` where every run collapsed shows as NaN.

```python
search = choose_components(y, 6, seed=1)
print(search.k, np.round(search.bic).tolist())
```

```text
3 [9711.0, 8011.0, 7191.0, 7219.0, 7247.0, 7275.0]
```

The usual proof of BIC needs conditions that do not hold for mixtures, because a model with fewer
components sits on the edge of one with more. Keribin (2000) shows that it still finds the true
number when the components are Gaussian. When they are not, BIC adds components to describe the
shape. One skewed cloud is no mixture, and BIC can still say there are several groups:

```python
skewed = np.exp(0.9 * rng.standard_normal((1500, 2)))
print(choose_components(skewed, 5, seed=1).k)
```

```text
5
```

A pick above the number of groups you expect is a sign that the groups are not Gaussian. Look at the
data before you read it as more groups.

## Groups of any shape: DBSCAN

`dbscan(x, eps)` follows Ester, Kriegel, Sander and Xu (1996). A point is a core point when at least
`min_samples` points lie within distance `eps` of it, the point itself included. A cluster is a set
of core points joined by steps of at most `eps`, plus the border points within `eps` of one of its
core points. Every other point is noise, labelled -1. There is no number of clusters to give.

A border point within `eps` of core points of two clusters joins the cluster discovered first
(section 4.1 of the paper). Here that is the one whose lowest-index core point comes first, so the
result does not depend on how you order the rows, apart from that choice.

```python
from statsx.clustering import dbscan, hdbscan, k_distances

t = rng.uniform(0.0, np.pi, 300)
upper = np.column_stack([np.cos(t), np.sin(t)])
lower = np.column_stack([1.0 - np.cos(t), 0.4 - np.sin(t)])
moons = np.vstack([upper, lower]) + 0.05 * rng.standard_normal((600, 2))
moon_truth = np.repeat([0, 1], 300)

db = dbscan(moons, 0.2, min_samples=5)
print(db.n_clusters, db.n_noise, db.n_border, db.sizes.tolist())
km = kmeans(moons, 2, seed=1)
def agreement(labels):
    share = float((labels == moon_truth).mean())
    return round(max(share, 1.0 - share), 2)

print(agreement(db.labels), agreement(km.labels))
```

```text
2 0 0 [300, 300]
1.0 0.8
```

The two half-circles interleave. DBSCAN follows each one. k-means cuts the plane in two with a line.

### Choosing eps

`eps` has no default, because it sets what counts as dense in your units. Ester et al. (section 4.2)
sort the distance of every point to its k-th neighbour, largest first, and take `eps` at the first
valley of that curve. `k_distances(x, k)` gives those distances. With `min_samples = k + 1`, a point
is a core point exactly when its k-distance is at most `eps`. The paper uses `k = 4`.

```python
curve = np.sort(k_distances(moons, 4))[::-1]
print(np.round(curve[[0, 10, 60, 300, 599]], 3).tolist())
```

```text
[0.131, 0.108, 0.076, 0.048, 0.018]
```

## Groups of different density: HDBSCAN

One `eps` cannot suit a tight group and a spread-out one. A radius small enough to split two tight
groups leaves the spread-out one as noise. A radius large enough to hold it joins the tight groups.
`hdbscan(x, min_cluster_size)` removes the choice. It follows Campello, Moulavi and Sander (2013)
and Campello, Moulavi, Zimek and Sander (2015), in the notation of McInnes and Healy (2017, section
2.2):

1. The core distance of a point is its distance to its `min_samples`-th nearest point, the point
   itself being the first.
2. The mutual reachability distance of two points is the largest of their two core distances and
   their distance.
3. The single linkage hierarchy of that distance is read from a minimum spanning tree.
4. The hierarchy is condensed. A branch with fewer than `min_cluster_size` points is dropped, and
   its points fall out of the parent at density `lambda = 1 / distance`.
5. The stability of a cluster is the sum, over its points, of the density they leave at minus the
   density the cluster is born at.
6. The result is the set of non-overlapping clusters with the largest total stability. A point in no
   chosen cluster is noise.

The tree is exact. It is built by Prim's algorithm from distances, with no neighbour index, so the
cost is O(n^2 d) time. Edges of equal weight, which mutual reachability makes common, join at one
level, so the result does not depend on the order of equal edges.

Memory is O(n d), except for up to 8,192 rows of 64 columns or more: then every pair distance is
computed once and kept, 4 n^2 bytes (100 MB at 5,000 rows, 268 MB at 8,192). On 1,024 columns and
4 threads this made `hdbscan` 2.4 to 3.6 times faster, with the same result: 4.2 s against 9.9 s
at 5,000 rows.

```python
dense = rng.standard_normal((90, 2)) * 0.2
more = rng.standard_normal((90, 2)) * 0.2 + np.array([1.5, 0.0])
wide = rng.standard_normal((90, 2)) * 1.6 + np.array([10.0, 8.0])
scatter = rng.uniform(-4, 14, (25, 2))
z = np.vstack([dense, more, wide, scatter])

hd = hdbscan(z, 15)
print(hd.n_clusters, hd.n_noise, hd.sizes.tolist())
print(np.round(hd.stabilities, 1).tolist())
```

```text
3 13 [102, 90, 90]
[60.3, 389.5, 408.4]
```

Three groups are found: two tight ones 1.5 apart and one spread out. Of the 25 scattered points, 13
are noise and 12 fall inside the spread-out group, which has 102 points. `hd.stabilities` is the
stability of each chosen cluster. `hd.tree` is the condensed tree, and `hd.mst` is the minimum
spanning tree with one row per edge.

```python
for eps in (0.3, 0.5, 1.0, 1.5):
    fit = dbscan(z, eps, min_samples=15)
    print(eps, fit.n_clusters, fit.n_noise, fit.sizes.tolist())
```

```text
0.3 2 117 [89, 89]
0.5 2 115 [90, 90]
1.0 2 54 [180, 61]
1.5 2 24 [180, 91]
```

With the same `min_samples`, no radius finds all three. At 0.3 and 0.5 DBSCAN finds the two tight
groups and loses the spread-out one as noise. At 1.0 and 1.5 it joins the two tight groups into one
cluster.

### One group, or none

The root of the tree is the whole data set. It is not a candidate unless
`allow_single_cluster=True`. So HDBSCAN answers "which groups stand out from each other", not "is
there a group here". This default is a choice of statsx. The papers do not state it.

HDBSCAN returns clusters for data that have none, when `min_cluster_size` is small for the number of
points. On 300 samples of one Gaussian cloud of 200 points, `min_cluster_size = 10` gave a cluster
on 192, `min_cluster_size = 20` on 2, and `min_cluster_size = 30` on none. A result is not evidence
of structure. Check that the clusters hold when you resample the rows and change `min_cluster_size`.

```python
cloud = np.random.default_rng(0).standard_normal((200, 2))
print(hdbscan(cloud, 20).n_clusters, hdbscan(cloud, 20, allow_single_cluster=True).n_clusters)
```

```text
0 1
```

`min_cluster_size` is the smallest group you accept as a cluster. `min_samples` (default
`min_cluster_size`) sets how smooth the density estimate is: a larger value turns more points into
noise.

## What it assumes

- k-means assumes round clusters of similar spread and size. It minimizes a sum of squares, which is
  the maximum likelihood partition under those conditions.
- A Gaussian mixture assumes each group is Gaussian. A group that is skewed or has heavy tails is
  split into several components, and BIC supports the split. Its density can look fine while the
  groups are not real.
- DBSCAN and HDBSCAN assume that closeness in your space means similarity, and that a group is a
  region where points are denser than around it. A group with a gradual edge has no clear border:
  the result changes with `eps` or `min_samples`.
- Distances mean something in your space. Euclidean distance on unscaled columns is dominated by the
  column with the largest range. For text embeddings, divide each row by its norm first.
- The gap statistic assumes the null model "uniform over the range of the data". Data that are not
  uniform and not clustered, such as one long skewed cloud, may get a pick above 1.

## What the library checks and does not check

Checked: the points are a finite (n, d) array with at least 2 rows, `k` is from 1 to n, and the
counts and tolerances are in range. A fit reports `converged`, and says False when a run reached
`max_iter`. A mixture run that ends with a collapsed component is dropped and counted in
`n_degenerate`.

Not checked: that a noise label means an outlier. It says the point is not in a dense enough
region at these settings. Change `min_samples` or `min_cluster_size` and the same point can join a
cluster.

Not checked: that the groups are Gaussian. A mixture that converged says the likelihood stopped
rising, not that the data came from Gaussians.

Not checked: that the clusters exist. k-means returns `k` clusters for any data, including one
Gaussian cloud, so a result is no evidence of structure. Read the silhouette and the gap. Neither
proves that the clusters are real: they say how well separated the partition is.

## Checked in the tests

- The fit reaches the best partition, found by trying all assignments, on 10 points with 3 clusters.
- The result is identical with 1 and with 4 threads, bit for bit.
- The silhouette equals its definition computed by hand on random data.
- With `metric="cosine"`, k-means reaches the best partition found by trying all assignments on
  10 points, its centres are the unit means and its labels the largest cosine. DBSCAN equals the
  sequential algorithm and HDBSCAN the hierarchy read from the definitions, both on the `1 - cos`
  matrix, on 4 random data sets and 3 settings. `k_distances` equals the same matrix. Scaling the
  rows changes nothing, and the fit is identical with 1 and 4 threads.
- `choose_k` picks the first `k` whose margin is 0 or more, and `decision_margin` is the smallest
  deciding margin in standard errors. On the flat curve above the pick changes with
  `n_reference` and every run has a margin below 1. With `k_min` the curve is the same, cut off.
- `dispersion` equals the mean distance computed cluster by cluster, in both distances.
- `assign_clusters` gives the nearest centre and its distance as computed from all pairs, in both
  distances; a row at the limit keeps its label, and a tie goes to the first centre.
- `kmeans`, the assignment and the silhouette give the same result for data moved to 1e8 and 1e11
  from the origin.
- The assignment equals the smallest distance over all centroids, label for label, on data with many
  ties, far from the origin, beyond the range of float32 and on 1,100 columns, and its bound never
  rules out that centroid.
- The mini-batch kernel gives the bytes of Algorithm 1 written out in a test, with the same draws.
  `minibatch_kmeans` gives the same bytes for 1 and 3 threads and for float32, each row gets its
  nearest centre, and on 8 separate groups the inertia is within 2% of `kmeans`.
- float32 input gives the same bytes as its float64 copy, and a fit and an assignment on 200,000
  rows of 256 columns raise the peak memory by less than half the size of the data.
- `find_knees` reproduces Figure 2 of the Kneedle paper (the knee at `x = 0.22`) and equals the
  steps of the paper, written out in the test, on 300 random curves and 3 sensitivities.
- `elbow_k` found 3, 5 and 7 planted clusters for each of 3 seeds, and picked a `k` for uniform
  data, where `choose_k` answered 1.
- Three planted blobs gave `k = 3` in 40 of 40 runs of `choose_k`. Uniform data gave `k = 1` in 38
  of 40 runs, and `k = 2` in the others.
- The mixture log-likelihood never decreases, for full, diagonal and spherical covariances.
- The log-likelihood equals the log of the mixture density summed over the points, computed by hand.
- A mixture with one component equals the sample mean and covariance.
- On Gaussian data BIC gave the true number of components in 7 or more of 8 runs. On one skewed
  cloud it gave more than 1.
- The fit is identical with 1 and with 4 threads, bit for bit.
- DBSCAN equals the sequential algorithm of Ester et al. (1996), written out in the test, label for
  label, on random data, on an integer grid where points sit exactly at `eps`, on points far from
  the origin, and in 30 dimensions.
- `k_distances` and the core distances are the `k`-th smallest of the distances to all rows, bit
  for bit, also far from the origin and when many distances tie within a rounding unit. With the
  pair distances kept or computed as needed, the tree of HDBSCAN is the same, bit for bit.
- The core distances and the tree weight of HDBSCAN equal the definitions, computed on the full
  distance matrix. The clusters and their stabilities equal a second implementation that reads the
  hierarchy from the top, by removing the heaviest edges, on 6 random data sets and 3 settings.
- Two interleaved half-circles are separated by DBSCAN and not by k-means. On data with two tight
  groups and one spread-out group, HDBSCAN found all three on 3 of 3 seeds and no `eps` found more
  than two.
- On 300 samples of one Gaussian cloud of 200 points, HDBSCAN gave a cluster on 192 with
  `min_cluster_size = 10`, on 2 with 20, and on none with 30.
- DBSCAN and HDBSCAN are identical with 1 and with 4 threads, bit for bit.
- 100,000 points in 8 dimensions took about 42 s for HDBSCAN and 27 s for DBSCAN on 4 threads, on a
  machine in use.
- 200,000 points in 32 dimensions with `k = 20` took 4.8 s for 300 iterations on 4 threads.

## References

- Sculley, D. (2010). Web-scale k-means clustering. *Proceedings of the 19th International
  Conference on World Wide Web*, pp. 1177-1178. <https://doi.org/10.1145/1772690.1772862>
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
  <https://doi.org/10.1137/1.9780898718027>. Section 3.1 gives the error bound of an inner product.
- Dhillon, I. S., and Modha, D. S. (2001). Concept decompositions for large sparse text data using
  clustering. *Machine Learning* 42(1-2):143-175. <https://doi.org/10.1023/A:1007612920971>

- Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). Finding a "Kneedle" in a haystack:
  detecting knee points in system behavior. *31st International Conference on Distributed
  Computing Systems Workshops*, pp. 166-171. <https://doi.org/10.1109/ICDCSW.2011.20>

- Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). A density-based algorithm for
  discovering clusters in large spatial databases with noise. *Proceedings of the 2nd International
  Conference on Knowledge Discovery and Data Mining (KDD-96)*, pp. 226-231.
- Campello, R. J. G. B., Moulavi, D., and Sander, J. (2013). Density-based clustering based on
  hierarchical density estimates. *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*,
  LNCS 7819, pp. 160-172. <https://doi.org/10.1007/978-3-642-37456-2_14>
- Campello, R. J. G. B., Moulavi, D., Zimek, A., and Sander, J. (2015). Hierarchical density
  estimates for data clustering, visualization, and outlier detection. *ACM Transactions on
  Knowledge Discovery from Data* 10(1), article 5. <https://doi.org/10.1145/2733381>
- McInnes, L., and Healy, J. (2017). Accelerated hierarchical density based clustering. *2017 IEEE
  International Conference on Data Mining Workshops (ICDMW)*, pp. 33-42.
  <https://doi.org/10.1109/ICDMW.2017.12>
- Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data
  via the EM algorithm. *Journal of the Royal Statistical Society B* 39(1):1-38.
  <https://doi.org/10.1111/j.2517-6161.1977.tb01600.x>
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics* 6(2):461-464.
  <https://doi.org/10.1214/aos/1176344136>
- McLachlan, G. J., and Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Keribin, C. (2000). Consistent estimation of the order of mixture models. *Sankhya A* 62(1):49-66.
- Fraley, C., and Raftery, A. E. (2002). Model-based clustering, discriminant analysis, and density
  estimation. *Journal of the American Statistical Association* 97(458):611-631.
  <https://doi.org/10.1198/016214502760047131>

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*
  28(2):129-137. <https://doi.org/10.1109/TIT.1982.1056489>
- Arthur, D., and Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding.
  *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, pp. 1027-1035.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of
  cluster analysis. *Journal of Computational and Applied Mathematics* 20:53-65.
  <https://doi.org/10.1016/0377-0427(87)90125-7>
- Tibshirani, R., Walther, G., and Hastie, T. (2001). Estimating the number of clusters in a data
  set via the gap statistic. *Journal of the Royal Statistical Society B* 63(2):411-423.
  <https://doi.org/10.1111/1467-9868.00293>
