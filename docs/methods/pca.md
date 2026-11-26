# Reduce many columns: PCA

Use this when each point has many columns, such as an embedding, and you want fewer before you
cluster or map them. The function is `pca` in `statsx.embedding`. It follows Pearson (1901) and
Hotelling (1933).

```python
import numpy as np
from statsx.clustering import elbow_k
from statsx.embedding import pca

rng = np.random.default_rng(0)
centres = rng.normal(size=(8, 256))
centres /= np.linalg.norm(centres, axis=1, keepdims=True)
members = rng.integers(0, 8, size=2000)
x = centres[members] + 0.2 * rng.normal(size=(2000, 256))
x /= np.linalg.norm(x, axis=1, keepdims=True)

reduced = pca(x, 8)
elbow = elbow_k(reduced.scores, 14)
print(reduced.scores.shape, elbow.k, elbow.is_knee)
```

```text
(2000, 8) 8 True
```

The data are 8 groups in 256 columns, with noise much longer than the groups. On the 8 first
components the elbow finds the 8 groups.

## What you get

`pca(x, k)` returns a `PCAFit`:

- `scores`: the coordinates of each row on the components, shape (n, k).
- `components`: the directions, largest variance first.
- `explained_variance` and `explained_variance_ratio`: how much of the variance each one holds.
- `transform(new_rows)`: the same projection for rows that arrive later.

The projection is the linear one that loses the least: the sum of squared differences between the
rows and their reconstruction is the smallest possible for `k` directions.

## Use it with clustering and UMAP

1. Cosine distance: scale each row to length 1 first, then reduce, then cluster with the Euclidean
   distance.
2. Centres in the original columns: cluster the scores, then average the rows of `x` by label.
3. UMAP on wide data: `umap(pca(x, 100).scores)`. Place new rows with
   `fit.transform(pca_fit.transform(new_rows))`.

## Choose the number of components

Look at `explained_variance_ratio` of a larger fit and see where it levels off. For clustering, try
two or three values near the number of groups you expect, and check that the result holds. No rule
here chooses the number for you.

## What it assumes

PCA keeps the directions of largest variance. Groups that differ along a direction of small
variance are lost. Here two groups differ in the third column, which varies least:

```python
group = np.repeat([0, 1], 200)
flat = rng.normal(size=(400, 3)) * [20.0, 10.0, 0.1]
flat[:, 2] += group
kept = pca(flat, 2)
print(np.abs(kept.components[:, 2]).max() < 0.1, kept.explained_variance_ratio.sum() > 0.99)
```

```text
True True
```

The two components hold more than 99% of the variance and almost nothing of the third column, so no
clustering of the scores can find the groups.

PCA also does not scale the columns. If they have different units, divide each by its standard
deviation first, or the column with the largest numbers decides the components.

## What the library checks and does not check

Checked: the data are finite, `k` is from 1 to `min(n - 1, d)`, and the rows are not all equal.

Not checked: that the components keep what you care about. A high share of variance does not say
that the groups survived.

## Cost

The components are exact. With `d <= n` the cost is O(n d^2 + d^3), and the data are read in blocks
of 4,096 rows, so no second copy is made. NumPy's linear algebra library decides how many threads
the products use. When two eigenvalues are equal, another machine can return another pair of
directions in their plane.

## Checked in the tests

- The components equal the singular vectors of the centred data, and the variances its squared
  singular values, with more rows than columns, with more columns than rows, and over several
  blocks of rows.
- The reconstruction error equals the sum of the left-out squared singular values.
- The entry of largest size in each component is positive, and `transform` of the data returns the
  scores.
- On 2,000 rows of 256 columns with 8 groups, the elbow on 8 components gives 8, with more than 90%
  of the rows in their group.

## References

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.
  *Philosophical Magazine* 2(11):559-572. <https://doi.org/10.1080/14786440109462720>
- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
  *Journal of Educational Psychology* 24(6):417-441. <https://doi.org/10.1037/h0071325>
