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
| `model.to_npz`, `UMAPModel.from_npz` | `fit.to_npz`, `UMAPFit.from_npz` |
| `kmeans(x, k)` | `statsx.clustering.kmeans(x, k)` |
| `metric="cosine"` (clustering) | the same argument of `kmeans`, `dbscan`, `hdbscan`, `elbow_k` |
| `cluster(x, method="kmeans", n_clusters=k)` | `kmeans(x, k).labels` |
| `cluster(x, method="hdbscan")` | `hdbscan(x).labels` |
| `cluster(x)` with no number of clusters | `kmeans(x, elbow_k(x).k).labels` |
| `assign_clusters(new, centroids)` after a statsx fit | `fit.predict(new)` |
| `dbscan(x, eps=...)` | `statsx.clustering.dbscan(x, eps)` |
| `hdbscan(x, min_cluster_size=...)` | `statsx.clustering.hdbscan(x, min_cluster_size)` |
| `neighbor_overlap(x, y, k)` | `neighbor_preservation(x, y, k)` |
| `choose_k(x)` (the elbow) | `elbow_k(x).k` |
| `choose_k(x, return_evidence=True)` | `elbow_k(x)`: `ks`, `inertia`, `inertia_rises` |

Files saved by `fastumap` do not load in `statsx`. Fit again and save.

## What changes in the results

The maps and the labels are not the same numbers as in `fastumap`. Check what you read from them
again after the switch.

- UMAP: other random draws and exact neighbours, so another map of the same data.
- k-means: full Lloyd iterations in float64 with 10 starts. `fastumap` uses one start by default.
- HDBSCAN and DBSCAN: exact distances over all pairs. `fastumap` uses a graph of 30 neighbours.
- Cosine k-means is spherical k-means: the centres have length 1. In `fastumap` a centre is the
  mean of the unit rows, which is shorter, so some labels differ.
- Cosine HDBSCAN uses the densities of `1 - cos`. `fastumap` uses the Euclidean distance of the
  unit rows, so it can choose other clusters. Cosine `eps` is in `1 - cos` in both.
- `eps` has no default in `dbscan`. Use `k_distances` to choose it.
- `choose_k` means another thing: in `statsx` it is the gap statistic. The elbow is `elbow_k`.
  It takes no `max_samples`: sample the rows first if you need to. If you use the gap statistic,
  read `decision_margin`: on embeddings the curve can be flat.
- `transform`: see [Place new points in a map](methods/embedding.md#place-new-points-in-a-map).

## Planned

In this order.

| Need | fastumap | Status in statsx |
|---|---|---|
| MG-3 | `assign_clusters` with your centroids, `max_distance` | `predict` of a statsx fit only |
| MG-4 | per-cluster `dispersion` | sizes only |
| MG-5 | speed and memory at 1024 columns, `batch_size`, `pca_dim` | not measured |
| MG-6 | thread cap in containers, thread safety | up to 4 threads from the CPU count of the host |
| MG-7 | `init=` | not there |
| MG-8 | `y=` (supervised) | not there |
| MG-9 | `knn=` (approximate), `precomputed_knn` | exact neighbours only |

Until MG-6 is done, pass `threads=1` inside a container with less than one CPU.

## Not planned

No issue of `fastumap` asked for these. Open an issue in `statsx` if you use one.

- `densmap`, `spread`, `initial_alpha`, `chunk_count`
- `spectral_project`, spectral clustering
- `global_distance_correlation`, `random_layout`, the `grade` command
