<h1 align="center">statsx</h1>

<p align="center"><strong>Statistics for experiments, model evaluations and forecasts, where every method has a page that states what it assumes.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/statsx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/statsx?style=flat-square"></a>
  <a href="https://pypi.org/project/statsx/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/statsx?style=flat-square"></a>
  <a href="https://gitlab.com/cardona-oss/statsx/-/pipelines"><img alt="Pipeline" src="https://img.shields.io/gitlab/pipeline-status/cardona-oss%2Fstatsx?branch=main&style=flat-square"></a>
  <a href="https://gitlab.com/cardona-oss/statsx/-/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="https://gitlab.com/cardona-oss/statsx/-/blob/main/docs/choose.md">Which method do I use?</a> ·
  <a href="https://gitlab.com/cardona-oss/statsx/-/tree/main/docs/methods">Methods</a> ·
  <a href="https://gitlab.com/cardona-oss/statsx/-/tree/main/docs/recipes">Recipes</a> ·
  <a href="https://gitlab.com/cardona-oss/statsx/-/blob/main/CHANGELOG.md">Changelog</a> ·
  <a href="https://gitlab.com/cardona-oss/statsx/-/issues">Issues</a>
</p>

---

## Overview

statsx covers intervals, Bayesian updates, tests, sequential tests, experiment design, model
comparison and forecasting of count series. Each method has a methods page with the model, the
formulas and what it assumes. Part of the numerical code is a Rust extension inside the wheel,
and NumPy is the only dependency.

Status: beta.

## Install

```bash
pip install statsx
```

Requires Python 3.12 or newer. NumPy is the only run-time dependency.

## Quickstart

An interval for a proportion, 7 successes in 40 trials:

```python
from statsx.proportion import proportion_interval

result = proportion_interval(7, 40, method="wilson")
print(result.estimate)
print(result.interval)
print(result.n)
print(result.assumption)
```

```text
0.175
Interval(lower=0.0874541374603592, upper=0.3194999033178771, level=0.95)
40
independent trials with one success probability; Wilson score interval
```

A function that returns an `Estimate`, such as this one, gives the estimate, the interval, the
sample size and an `assumption` field. Fitted models (`fit_trend`, `fit_bradley_terry`) and
functions that return a bare `Interval` do not have that field: their methods pages state what
they assume.

## What it does

Not sure where to start? Read
[Which method do I use?](https://gitlab.com/cardona-oss/statsx/-/blob/main/docs/choose.md).

Each method follows a published paper, and its module docstring lists the references. The
[methods pages](https://gitlab.com/cardona-oss/statsx/-/tree/main/docs/methods) state the model,
the formulas and the guarantee.

Of the 31 method modules, 19 import the Rust extension `statsx._accel`, 6 import it through
another statsx module, and 6 are Python and NumPy only (`allocation`, `alwaysvalid`,
`groupsequential`, `identify`, `multiple`, `rating`).

- Forecasting: trend with forecast and anomalies, causal impact, rolling-origin backtest.
- Duration models: estimated time to finish, from past durations and from progress.
- Intervals: proportions, bootstrap (percentile, basic, BCa), effect sizes, credible intervals.
- Bayes: conjugate updates, lognormal prior from an expert interval, P(A > B).
- Tests: multiple testing, equivalence (TOST), paired tests, agreement (kappa, alpha).
- Decisions: Monte Carlo cost, expected value of information, break-even.
- Calibration: reliability, Brier decomposition, Platt scaling, isotonic regression.
- Design: power, sample size, minimum detectable effect, group sequential bounds.
- Sequential: confidence sequences, e-values, Thompson sampling, best-arm identification.
- Distributions: fits from sufficient statistics, goodness of fit at an effective sample size.
- Model comparison: Rasch abilities of models on shared tasks, token ratios, cost per pass.
- Ranking: Bradley-Terry with ties, a fit per judge, Elo, Glicko-2.
- Drift: Kolmogorov-Smirnov, MMD, classifier two-sample test, PSI.
- MCMC checks: convergence, effective sample size, Monte Carlo standard error, simulation-based
  calibration.

## Links

- Source: https://gitlab.com/cardona-oss/statsx
- Issues: https://gitlab.com/cardona-oss/statsx/-/issues
- PyPI: https://pypi.org/project/statsx/
- Changelog: [CHANGELOG.md](https://gitlab.com/cardona-oss/statsx/-/blob/main/CHANGELOG.md)

## Contributing

Developer setup, the dev loop and the release process are in
[CONTRIBUTING.md](https://gitlab.com/cardona-oss/statsx/-/blob/main/CONTRIBUTING.md).
In short: `uv sync && make install-hooks`, then `make check` runs lint, format check, typecheck
and tests.

## License

MIT © 2026 Jorge Cardona (see [LICENSE](https://gitlab.com/cardona-oss/statsx/-/blob/main/LICENSE)).
