# statsx

[![pipeline status](https://gitlab.com/cardona-oss/statsx/badges/main/pipeline.svg)](https://gitlab.com/cardona-oss/statsx/-/pipelines)
[![PyPI version](https://img.shields.io/pypi/v/statsx.svg)](https://pypi.org/project/statsx/)
[![Python versions](https://img.shields.io/pypi/pyversions/statsx.svg)](https://pypi.org/project/statsx/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.com/cardona-oss/statsx/-/blob/main/LICENSE)

Statistics for Python with its own Rust core: Bayesian inference, forecasting, causal impact, tests and design, with NumPy as the only dependency.

## Install

```bash
pip install statsx
```

Requires Python 3.12 or newer. NumPy is the only run-time dependency.

## Quickstart

An interval for a proportion, 7 successes in 40 trials:

```python
from statsx.proportion import proportion_interval

print(proportion_interval(7, 40, method="wilson"))
```

```text
Estimate(estimate=0.175, interval=Interval(lower=0.0874541374603592, upper=0.3194999033178771, level=0.95), n=40, assumption='independent trials with one success probability; Wilson score interval')
```

Interval results such as this one carry the estimate, the interval, the sample size and an
`assumption` field. Fitted models (`fit_trend`, `fit_bradley_terry`) do not have that field: their
methods pages state what they assume.

## What it does

Not sure where to start? Read [Which method do I use?](https://cardona-oss.gitlab.io/statsx/choose/).

Each method is written in Rust from a published paper, and its docstring lists the references.
The [methods pages](https://cardona-oss.gitlab.io/statsx/methods/always-valid/) state the model,
the formulas and the guarantee.

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

## Links

- **Documentation:** https://cardona-oss.gitlab.io/statsx/
- **Source:** https://gitlab.com/cardona-oss/statsx
- **Issues:** https://gitlab.com/cardona-oss/statsx/-/issues
- **PyPI:** https://pypi.org/project/statsx/
- **Changelog:** [CHANGELOG.md](https://gitlab.com/cardona-oss/statsx/-/blob/main/CHANGELOG.md)

## Contributing

Developer setup, the dev loop, and the release process live in
[CONTRIBUTING.md](https://gitlab.com/cardona-oss/statsx/-/blob/main/CONTRIBUTING.md). In short: `uv sync && make install-hooks`, then
`make check` runs the same gate CI runs.

## License

MIT © 2026 Jorge Cardona (see [LICENSE](https://gitlab.com/cardona-oss/statsx/-/blob/main/LICENSE)).
