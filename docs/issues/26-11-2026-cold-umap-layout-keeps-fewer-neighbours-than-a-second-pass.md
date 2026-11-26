---
title: A cold UMAP fit keeps fewer neighbours than a second pass at step size 0.1
status: open
area: core
owner: null
filed: 2026-11-26
fixed_in: null
priority: low
scope: M
verify: probe
---

Find out why, before any change of the default schedule.

## Seen

600 points in 8 Gaussian groups of 64 columns, 3 data sets (D-78):

- cold fit, 500 epochs: `neighbor_preservation` 0.35 to 0.36
- refresh of a moving window at `learning_rate=0.1`: 0.385 to 0.389

## To measure

- Other data shapes and sizes.
- A cold fit followed by a refresh of the same data.
- Whether the spectral start or the end of the schedule causes the difference.

## Also not measured for `init`

- A chain of many refreshes: how far the map drifts from the first one.
- A start from another source (PCA scores, another library's map).
