# Move from fastumap to statsx

Use this page to see which `fastumap` calls have a replacement in `statsx` today. `statsx` will
replace `fastumap`. `fastumap` will be archived when the rows marked "planned" are done.

Both libraries are written for the same use: UMAP and clustering with no compiler at import time.
`statsx` needs only NumPy. `fastumap` also needs SciPy.

## Ready

| fastumap | statsx |
|---|---|
| `umap_project(x, 2)` | `umap(x).embedding` |
| `umap_project(x, 10)` | `umap(x, dimensions=10).embedding` |
| `metric="cosine"` (UMAP) | `umap(x, metric="cosine")` |
| `fit(x)`, `model.transform(new)` | `fit = umap(x)`, `fit.transform(new).embedding` |
| `return_distances=True` | `fit.transform(new).distances[:, 0]` |
| `umap_project(x, 2, init=previous)` | `umap(x, init=previous).embedding` |
| `initial_alpha=a` | `learning_rate=a` |
| `precomputed_knn=(indices, distances)` | `umap(x, neighbors=(indices, distances))` |
| `y=y`, `target_weight=t` | `umap(x, labels=y, label_weight=w)` |
| `model.to_npz`, `UMAPModel.from_npz` | `fit.to_npz`, `UMAPFit.from_npz` |
| `kmeans(x, k)` | `statsx.clustering.kmeans(x, k)` |
| `metric="cosine"` (clustering) | the same argument of `kmeans`, `dbscan`, `hdbscan`, `elbow_k` |
| `cluster(x, method="kmeans", n_clusters=k)` | `kmeans(x, k).labels` |
| `batch_size=b` (mini-batch) | `minibatch_kmeans(x, k, batch_size=b)` |
| `cluster(x, method="hdbscan")` | `hdbscan(x).labels` |
| `cluster(x)` with no number of clusters | `kmeans(x, elbow_k(x).k).labels` |
| `assign_clusters(new, centroids)` | `assign_clusters(new, centroids).labels` |
| `max_distance=r`, `return_distances=True` | the same argument; `.labels`, `.distances` |
| `ClusterResult.sizes`, `.dispersion` | `fit.sizes`, `fit.dispersion` of `kmeans` |
| `dbscan(x, eps=...)` | `statsx.clustering.dbscan(x, eps)` |
| `hdbscan(x, min_cluster_size=...)` | `statsx.clustering.hdbscan(x, min_cluster_size)` |
| `neighbor_overlap(x, y, k)` | `neighbor_preservation(x, y, k)` |
| `choose_k(x)` (the elbow) | `elbow_k(x).k` |
| `pca_dim=m` | `pca(x, m).scores` first, then call `kmeans`, `hdbscan` or `umap` on it |
| `choose_k(x, return_evidence=True)` | `elbow_k(x)`: `ks`, `inertia`, `inertia_rises` |

Files saved by `fastumap` do not load in `statsx`. Fit again and save.

## What changes in the results

The maps and the labels are not the same numbers as in `fastumap`. Check what you read from them
again after the switch.

- UMAP: other random draws and exact neighbours, so another map of the same data.
- k-means: full Lloyd iterations in float64 with 10 starts. `fastumap` uses one start by default.
  `fastumap` switches to mini-batches by itself on large inputs; `statsx` never does. Call
  `minibatch_kmeans` for that.
- HDBSCAN and DBSCAN: exact distances over all pairs. `fastumap` uses a graph of 30 neighbours.
- Cosine k-means is spherical k-means: the centres have length 1. In `fastumap` a centre is the
  mean of the unit rows, which is shorter, so some labels differ.
- Cosine HDBSCAN uses the densities of `1 - cos`. `fastumap` uses the Euclidean distance of the
  unit rows, so it can choose other clusters. Cosine `eps` is in `1 - cos` in both.
- `pca_dim` in `fastumap` returns centres in the original columns. With `pca`, average the rows by
  label yourself. For the cosine distance, scale the rows to length 1 before `pca`.
- Centres saved from a `fastumap` cosine fit: scale your rows to length 1 and call
  `assign_clusters` with the Euclidean distance. That is what `fastumap` does, and `max_distance`
  keeps its meaning. With `metric="cosine"` the limit is in `1 - cos`.
- `dispersion` of a cosine fit is the mean `1 - cos`. In `fastumap` it is a Euclidean distance of
  unit rows, so its numbers are larger.
- `eps` has no default in `dbscan`. Use `k_distances` to choose it.
- `choose_k` means another thing: in `statsx` it is the gap statistic. The elbow is `elbow_k`.
  It takes no `max_samples`: sample the rows first if you need to. If you use the gap statistic,
  read `decision_margin`: on embeddings the curve can be flat.
- `init`: the step size starts at 0.1, not 1.0, so rows move less than in `fastumap` at its
  default. `init` is never written to. It needs one position for every row. See
  [Refresh a map](methods/embedding.md#refresh-a-map).
- Neighbours: `statsx` always searches exactly. For many rows of many columns, build a table
  with an approximate library and pass it as `neighbors`; see
  [Bring your own neighbours](methods/embedding.md#bring-your-own-neighbours). Pass Euclidean
  distances (the square root of what faiss returns for L2) or only the row numbers.
- Labels: start from `label_weight = 1 - exp(-2.5 / (1 - target_weight))`, so 0.5 becomes 0.993.
  The maps differ: `fastumap` merges the two directions of the graph a second time, so even one
  class changes its map. `statsx` does not, and a row labelled -1 keeps its weights. See
  [Let labels guide a map](methods/embedding.md#let-labels-guide-a-map).
- `transform`: see [Place new points in a map](methods/embedding.md#place-new-points-in-a-map).

## Speed and memory

`statsx` is faster for UMAP and when you reduce the columns first. It is slower for k-means and
HDBSCAN on all 1024 columns. Use `pca` first for wide data.

| Call, 1024 columns, 4 threads | fastumap | statsx |
|---|---|---|
| UMAP, 1,000 rows | 4.0 s | 1.1 s |
| UMAP, 5,000 rows | 24 s | 11 s |
| PCA to 16, then HDBSCAN, 20,000 rows | 78 s | 30 s |
| PCA to 16, then k-means (k=40), 250,000 rows | 52 s, 5.1 GB | 33 s, 3.3 GB |
| HDBSCAN, 5,000 rows | 3.2 s | 17 s |
| elbow, k up to 10, 5,000 rows | 6.8 s | 15 s |
| k-means (k=40), 100,000 rows | 35 s, 2.0 GB | 64 s, 1.4 GB |
| k-means (k=40), 250,000 rows | 27 s, 4.4 GB | 205 s, 3.3 GB |

Measured on 2026-11-26 with `fastumap` 0.3.3 and a `statsx` build with the changes of 0.1.77, on
8 separated Gaussian groups, one run each, with other work on the machine (load 8 to 18 on 16
cores). Read the ratios, not the seconds. The quality was the same: equal purity and k-means
inertia within 0.05%, and UMAP kept as many neighbours or a few more.

Why `statsx` is slower on all columns: HDBSCAN computes every pair exactly, where `fastumap` uses
30 neighbours. k-means runs full Lloyd iterations until it converges, where `fastumap` uses
mini-batches for large inputs. `max_iter` limits the iterations of `kmeans`.

With half a CPU, UMAP on 5,000 rows took 37 to 45 s in `statsx` and 47 s in `fastumap`.

## Planned

In this order.

| Need | fastumap | Status in statsx |
|---|---|---|
| MG-5 | speed on wide data | measured below; being improved |
| MG-9 | `knn="approx"` or `"auto"` | exact search, or bring a table with `neighbors=` |

## Containers

`statsx` reads the CPU limit of a container and uses that many threads, at most 4. On a host that
hides the limit, such as AWS Fargate, set `STATSX_THREADS`. See
[Threads, containers and repeat runs](recipes/threads.md).

## Not planned

No issue of `fastumap` asked for these. Open an issue in `statsx` if you use one.

- `densmap`, `spread`, `chunk_count`
- `spectral_project`, spectral clustering
- `global_distance_correlation`, `random_layout`, the `grade` command
