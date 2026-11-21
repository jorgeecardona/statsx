# Cluster points: k-means, Gaussian mixtures and how many clusters

Use this when you have points with no labels (measurements, embeddings) and want to split them into
groups, check the groups, and choose how many there are. The module is `statsx.clustering`.

Use k-means for round groups of similar size. Use a Gaussian mixture when groups are stretched,
differ in spread, or overlap, or when you want a probability for each point.

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

## What it assumes

- k-means assumes round clusters of similar spread and size. It minimizes a sum of squares, which is
  the maximum likelihood partition under those conditions.
- A Gaussian mixture assumes each group is Gaussian. A group that is skewed or has heavy tails is
  split into several components, and BIC supports the split. Its density can look fine while the
  groups are not real.
- Distances mean something in your space. Euclidean distance on unscaled columns is dominated by the
  column with the largest range. For text embeddings, divide each row by its norm first.
- The gap statistic assumes the null model "uniform over the range of the data". Data that are not
  uniform and not clustered, such as one long skewed cloud, may get a pick above 1.

## What the library checks and does not check

Checked: the points are a finite (n, d) array with at least 2 rows, `k` is from 1 to n, and the
counts and tolerances are in range. A fit reports `converged`, and says False when a run reached
`max_iter`. A mixture run that ends with a collapsed component is dropped and counted in
`n_degenerate`.

Not checked: that the groups are Gaussian. A mixture that converged says the likelihood stopped
rising, not that the data came from Gaussians.

Not checked: that the clusters exist. k-means returns `k` clusters for any data, including one
Gaussian cloud, so a result is no evidence of structure. Read the silhouette and the gap. Neither
proves that the clusters are real: they say how well separated the partition is.

## Checked in the tests

- The fit reaches the best partition, found by trying all assignments, on 10 points with 3 clusters.
- The result is identical with 1 and with 4 threads, bit for bit.
- The silhouette equals its definition computed by hand on random data.
- Three planted blobs gave `k = 3` in 40 of 40 runs of `choose_k`. Uniform data gave `k = 1` in 38 of
  40 runs, and `k = 2` in the others.
- The mixture log-likelihood never decreases, for full, diagonal and spherical covariances.
- The log-likelihood equals the log of the mixture density summed over the points, computed by hand.
- A mixture with one component equals the sample mean and covariance.
- On Gaussian data BIC gave the true number of components in 7 or more of 8 runs. On one skewed
  cloud it gave more than 1.
- The fit is identical with 1 and with 4 threads, bit for bit.
- 200,000 points in 32 dimensions with `k = 20` took 4.8 s for 300 iterations on 4 threads.

## References

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
- Tibshirani, R., Walther, G., and Hastie, T. (2001). Estimating the number of clusters in a data set
  via the gap statistic. *Journal of the Royal Statistical Society B* 63(2):411-423.
  <https://doi.org/10.1111/1467-9868.00293>
