---
title: Make the exact neighbour search faster, and prove which rows it cuts
status: open
area: core
owner: null
filed: 2026-11-26
fixed_in: null
priority: medium
scope: M
verify: test
---

`umap_knn` takes 13 minutes for 100,000 rows of 1,024 columns on 4 threads of a loaded laptop
(D-79).

## Speed

Measured on one thread, in units of 1e9 operations per second, for 1,024 columns:

- the current float64 Gram blocks: 8 to 15
- a float32 product: 32 to 35
- a symmetric float64 product (each pair once): 25 to 29

Plan: screen the candidates in float32, recompute them in float64, and use each pair once for
both rows. An estimate, not measured together: about 3 times faster.

## The cut

`search` keeps the k best rows by the value of `|a|^2 + |b|^2 - 2 a.b` and recomputes only those.
Two rows whose distances differ by less than the rounding of that value can fall on the wrong
side of the k-th place. No failing case was found. A float32 screen needs a proof step anyway:
compare the cut with the rounding bound of the inner product, and search a row again in float64
when the proof fails.

## Tests

The same bytes as a brute-force float64 search: copies of rows, near ties at the k-th place, rows
far from the origin, 1 and 3 threads.

## After it

Measure NN-descent (Dong, Charikar and Li 2011) against the faster search on real embeddings of
1,024 columns, recall with `neighbor_recall`, before writing it (D-79).
